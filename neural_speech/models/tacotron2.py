import tensorflow as tf
from tensorflow.contrib.rnn import MultiRNNCell, OutputProjectionWrapper, LSTMBlockCell
from tensorflow.contrib.seq2seq import BasicDecoder

from models.utils.helpers import TacoTestHelper, TacoTrainingHelper
from models.utils.modules import embedding, attention_decoder, conv_and_lstm, postnet
from text.symbols import symbols
from util.infolog import log


class Tacotron2():
    def __init__(self, hparams):
        self._hparams = hparams

    def initialize(self, text_inputs, input_lengths, speaker_ids, mel_targets=None, linear_targets=None):
        '''Initializes the model for inference.

        Sets "mel_outputs", "linear_outputs", and "alignments" fields.

        Args:
          text_inputs: int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
            steps in the input time series, and values are character IDs
          input_lengths: int32 Tensor with shape [N] where N is batch size and values are the lengths
            of each sequence in inputs.
          speaker_ids: int32 Tensor containing ids of specific speakers
          mel_targets: float32 Tensor with shape [N, T_out, M] where N is batch size, T_out is number
            of steps in the output time series, M is num_mels, and values are entries in the mel
            spectrogram. Only needed for training.
          linear_targets: float32 Tensor with shape [N, T_out, F] where N is batch_size, T_out is number
            of steps in the output time series, F is num_freq, and values are entries in the linear
            spectrogram. Only needed for training.
        '''
        with tf.variable_scope('inference'):
            is_training = linear_targets is not None
            batch_size = tf.shape(text_inputs)[0]
            hp = self._hparams
            vocab_size = len(symbols)
            embedded_inputs = embedding(text_inputs, vocab_size, hp.embedding_dim)  # [N, T_in, embd_size]

            # extract speaker embedding if multi-speaker
            with tf.variable_scope('speaker'):
                if hp.num_speakers > 1:
                    speaker_embedding = tf.get_variable('speaker_embed',
                                                        shape=(hp.num_speakers, hp.speaker_embed_dim),
                                                        dtype=tf.float32)
                    # TODO: what about special initializer=tf.truncated_normal_initializer(stddev=0.5)?
                    speaker_embd = tf.nn.embedding_lookup(speaker_embedding, speaker_ids)
                else:
                    speaker_embd = None

            # Encoder
            encoder_outputs = conv_and_lstm(
                    embedded_inputs,
                    input_lengths,
                    conv_layers=hp.encoder_conv_layers,
                    conv_width=hp.encoder_conv_width,
                    conv_channels=hp.encoder_conv_channels,
                    lstm_units=hp.encoder_lstm_units,
                    is_training=is_training,
                    scope='encoder')  # [N, T_in, 512]

            # Attention Mechanism
            attention_cell = attention_decoder(encoder_outputs, hp.attention_dim, input_lengths, is_training,
                                               speaker_embd=speaker_embd, attention_type="location_sensitive")

            # Decoder (layers specified bottom to top):
            decoder_cell = MultiRNNCell([
                attention_cell,
                LSTMBlockCell(hp.decoder_lstm_units),
                LSTMBlockCell(hp.decoder_lstm_units)], state_is_tuple=True)  # [N, T_in, 1024]

            # Project onto r mel spectrograms (predict r outputs at each RNN step):
            output_cell = OutputProjectionWrapper(decoder_cell, hp.num_mels * hp.outputs_per_step)
            decoder_init_state = output_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            if is_training:
                helper = TacoTrainingHelper(text_inputs, mel_targets, hp.num_mels, hp.outputs_per_step)
            else:
                helper = TacoTestHelper(batch_size, hp.num_mels, hp.outputs_per_step)

            (decoder_outputs, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    BasicDecoder(output_cell, helper, decoder_init_state),
                    maximum_iterations=hp.max_iters)  # [N, T_out/r, M*r]

            # Reshape outputs to be one output per entry
            decoder_outputs = tf.reshape(decoder_outputs, [batch_size, -1, hp.num_mels])  # [N, T_out, M]

            # Postnet: predicts a residual
            postnet_outputs = postnet(
                    decoder_outputs,
                    layers=hp.postnet_conv_layers,
                    conv_width=hp.postnet_conv_width,
                    channels=hp.postnet_conv_channels,
                    is_training=is_training)
            mel_outputs = decoder_outputs + postnet_outputs

            # Convert to linear using a similar architecture as the encoder:
            expand_outputs = conv_and_lstm(
                    mel_outputs,
                    None,
                    conv_layers=hp.expand_conv_layers,
                    conv_width=hp.expand_conv_width,
                    conv_channels=hp.expand_conv_channels,
                    lstm_units=hp.expand_lstm_units,
                    is_training=is_training,
                    scope='expand')  # [N, T_in, 512]
            linear_outputs = tf.layers.dense(expand_outputs, hp.num_freq)  # [N, T_out, F]

            # Grab alignments from the final decoder state:
            # TODO: seems not to work?!?
            alignments = tf.transpose(final_decoder_state[0].alignment_history.stack(), [1, 2, 0])

            self.inputs = text_inputs
            self.input_lengths = input_lengths
            self.mel_outputs = mel_outputs
            self.linear_outputs = linear_outputs
            self.alignments = alignments
            self.mel_targets = mel_targets
            self.linear_targets = linear_targets
            log('Initialized Tacotron model. Dimensions: ')
            log('  embedding:               %d' % embedded_inputs.shape[-1])
            log('  encoder out:             %d' % encoder_outputs.shape[-1])
            log('  concat attn & out:       %d' % attention_cell.output_size)
            log('  decoder cell out:        %d' % decoder_cell.output_size)
            log('  decoder out (%d frames):  %d' % (hp.outputs_per_step, decoder_outputs.shape[-1]))
            log('  decoder out (1 frame):   %d' % mel_outputs.shape[-1])
            log('  postnet out:             %d' % postnet_outputs.shape[-1])
            log('  linear out:              %d' % linear_outputs.shape[-1])

    def add_loss(self):
        '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
        with tf.variable_scope('loss'):
            hp = self._hparams
            self.mel_loss = tf.reduce_mean(tf.abs(self.mel_targets - self.mel_outputs))
            l1 = tf.abs(self.linear_targets - self.linear_outputs)
            # Prioritize loss for frequencies under 3000 Hz.
            n_priority_freq = int(2000 / (hp.sample_rate * 0.5) * hp.num_freq)
            self.linear_loss = 0.5 * tf.reduce_mean(l1) + 0.5 * tf.reduce_mean(l1[:, :, 0:n_priority_freq])
            self.loss = self.mel_loss + self.linear_loss

    def add_optimizer(self, global_step, gradient_clip=1.0):
        '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.

        Args:
          global_step: int32 scalar Tensor representing current global step in training
        '''
        with tf.variable_scope('optimizer'):
            hp = self._hparams
            # replace learning rate decay by exponential decay
            self.learning_rate = tf.train.exponential_decay(
                    hp.initial_learning_rate, global_step, hp.learning_rate_decay_halflife, 0.5)
            optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.adam_beta1, hp.adam_beta2)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            self.gradients = gradients
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip)

            # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
                                                          global_step=global_step)

    def add_stats(self):
        with tf.variable_scope('stats'):
            tf.summary.histogram('linear_outputs', self.linear_outputs)
            tf.summary.histogram('linear_targets', self.linear_targets)
            tf.summary.histogram('mel_outputs', self.mel_outputs)
            tf.summary.histogram('mel_targets', self.mel_targets)

            expected_mel = tf.transpose(tf.expand_dims(self.mel_outputs[:0], -1), [0, 2, 1, 3])
            inferred_mel = tf.transpose(tf.expand_dims(self.mel_targets[:0], -1), [0, 2, 1, 3])
            expected_spec = tf.expand_dims(self.linear_targets[:0], -1)
            inferred_spec = tf.expand_dims(self.linear_outputs[:0], -1)
            tf.summary.image("exp_mel", expected_mel * 255)
            tf.summary.image("got_mel", inferred_mel * 255)
            tf.summary.image("exp_spec", expected_spec * 255)
            tf.summary.image("got_spec", inferred_spec * 255)

            tf.summary.scalar('loss_mel', self.mel_loss)
            tf.summary.scalar('loss_linear', self.linear_loss)
            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('loss', self.loss)

            gradient_norms = [tf.norm(grad) for grad in self.gradients]
            tf.summary.histogram('gradient_norm', gradient_norms)
            tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))

            self.stats = tf.summary.merge_all()
