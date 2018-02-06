import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell

import models.utils.modules


class PrenetWrapper(RNNCell):
    '''Runs RNN inputs through a prenet before sending them to the cell.'''

    def __init__(self, cell, layer_sizes, is_training, speaker_embd=None):
        super(PrenetWrapper, self).__init__()
        self._cell = cell
        self._is_training = is_training
        self.layer_sizes = layer_sizes
        self._speaker_embd = speaker_embd

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def call(self, inputs, state):
        prenet_out = models.utils.modules.prenet(inputs, drop_rate=0.5, is_training=self._is_training,
                                                 layer_sizes=self.layer_sizes, scope="decoder_prenet", reuse=None)
        if self._speaker_embd is not None:
            s = tf.layers.dense(self._speaker_embd, prenet_out.shape[-1], activation=tf.nn.softsign)
            prenet_out = tf.concat([prenet_out, s], axis=-1, name="speaker_concat")
        return self._cell(prenet_out, state)

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)


class ConcatOutputAndAttentionWrapper(RNNCell):
    '''Concatenates RNN cell output with the attention context vector.

    This is expected to wrap a cell wrapped with an AttentionWrapper constructed with
    attention_layer_size=None and output_attention=False. Such a cell's state will include an
    "attention" field that is the context vector.
    '''

    def __init__(self, cell, speaker_embd=None):
        super(ConcatOutputAndAttentionWrapper, self).__init__()
        self._cell = cell
        self._speaker_embd = speaker_embd

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size + self._cell.state_size.attention

    def call(self, inputs, state):
        output, res_state = self._cell(inputs, state)
        out = [output, res_state.attention]
        if self._speaker_embd is not None:
            s = tf.layers.dense(self._speaker_embd, output.shape[-1], activation=tf.nn.softsign)
            out.append(s)
        return tf.concat(out, axis=-1, name="speaker_concat"), res_state

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)
