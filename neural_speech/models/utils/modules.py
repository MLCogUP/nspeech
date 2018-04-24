import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, LSTMBlockCell

import neural_speech.models.utils.rnn_wrappers
from neural_speech.models.utils.attention import LocationSensitiveAttention


def embedding(inputs, vocab_size, num_units, scope="embedding", reuse=None):
    '''
    Embeds a given tensor.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        embd = tf.get_variable('embedding',
                               [vocab_size, num_units],
                               dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                           stddev=0.01))  # stddev=0.5?
    return tf.nn.embedding_lookup(embd, inputs)  # [N, T_in, 256]


def prenet(inputs, drop_rate, is_training, layer_sizes, activation=tf.nn.relu, scope="prenet", reuse=None):
    x = inputs
    with tf.variable_scope(scope, reuse=reuse):
        for i, size in enumerate(layer_sizes):
            dense = tf.layers.dense(x, units=size, activation=activation, name='dense_%d' % (i + 1))
            x = tf.layers.dropout(dense, rate=drop_rate, name='dropout_%d' % (i + 1))
    return x


def conv_and_lstm(inputs, input_lengths, conv_layers, conv_width, conv_channels, lstm_units,
                  is_training, scope):
    # Convolutional layers
    with tf.variable_scope(scope):
        x = inputs
        for i in range(conv_layers):
            activation = tf.nn.relu if i < conv_layers - 1 else None
            x = conv1d(x, conv_width, conv_channels, activation, is_training, 'conv_%d' % i)

        # 2-layer bidirectional LSTM:
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            LSTMBlockCell(lstm_units),
            LSTMBlockCell(lstm_units),
            x,
            sequence_length=input_lengths,
            dtype=tf.float32,
            scope='encoder_lstm')

        # Concatentate forward and backwards:
        return tf.concat(outputs, axis=2)


def postnet(inputs, layers, conv_width, channels, is_training):
    x = inputs
    with tf.variable_scope('decoder_postnet'):
        for i in range(layers):
            activation = tf.nn.tanh if i < layers - 1 else None
            x = conv1d(x, conv_width, channels, activation, is_training, 'postnet_conv_%d' % i)
    return tf.layers.dense(x, inputs.shape[2])  # Project to input shape


def attention_decoder(inputs, num_units, input_lengths, is_training, speaker_embd=None, attention_type="bah",
                      scope="attention_decoder", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        if attention_type == 'bah_mon':
            attention_mechanism = tf.contrib.seq2seq.BahdanauMonotonicAttention(
                num_units, inputs)
        elif attention_type == 'bah_norm':
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units, inputs, normalize=True)
        elif attention_type == 'luong_scaled':
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units, inputs, scale=True)
        elif attention_type == 'luong':
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units, inputs)
        elif attention_type == 'bah':
            # Bahdanau et al. attention mechanism
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units,  # attention units
                inputs,
                memory_sequence_length=input_lengths
            )
        elif attention_type == "location_sensitive":
            attention_mechanism = LocationSensitiveAttention(num_units, inputs, memory_sequence_length=input_lengths)
        else:
            raise Exception("Unknown attention type ")

        # Attention
        if attention_type == "location_sensitive":
            pre_mechanism_cell = LSTMBlockCell(num_units)
        else:
            pre_mechanism_cell = GRUCell(num_units)

        # bottleneck prenet as in paper
        pre_mechanism = neural_speech.models.utils.rnn_wrappers.PrenetWrapper(pre_mechanism_cell, [256, 128],
                                                                              is_training,
                                                                              speaker_embd=speaker_embd)
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(
            pre_mechanism,  # 256
            attention_mechanism,  # 256
            alignment_history=True,
            output_attention=False)  # [N, T_in, 256]
        #  Concatenate attention context vector and RNN cell output into a 512D vector.
        concat_cell = neural_speech.models.utils.rnn_wrappers.ConcatOutputAndAttentionWrapper(
            attention_cell)  # [N, T_in, 512]
        return concat_cell


def conv1d_banks(inputs, K=16, activation=tf.nn.relu, is_training=True, scope="conv_bank", reuse=None):
    '''Applies a series of conv1d separately.

    Args:
      inputs: A 3d tensor with shape of [N, T, C]
      K: An int. The size of conv1d banks. That is,
        The `inputs` are convolved with K filters: 1, 2, ..., K.
      activation: a tf activation function
      is_training: A boolean. This is passed to an argument of `bn`.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A 3d tensor with shape of [N, T, K*Hp.embed_size//2].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Convolution bank: concatenate on the last axis to stack channels from all convolutions
        conv_outputs = tf.concat(
            [conv1d(inputs, k, 128, activation, is_training, 'conv1d_%d' % k) for k in range(1, K + 1)],
            axis=-1
        )
    return conv_outputs  # (N, T, Hp.embed_size//2*K)


def cbhg(inputs, input_lengths, activation=tf.nn.relu, speaker_embd=None, is_training=True,
         K=16, c=(128, 128), gru_units=128, num_highways=4, scope="cbhg"):
    with tf.variable_scope(scope):
        conv_bank = conv1d_banks(inputs, K=K, activation=activation, is_training=is_training)  # (N, T_x, K*E/2)

        # Maxpooling:
        conv_proj = tf.layers.max_pooling1d(conv_bank, pool_size=2, strides=1, padding='same')

        # Projection layers:
        for i, layer_size in enumerate(c[:-1]):
            conv_proj = conv1d(conv_bank, 3, layer_size, activation, is_training, 'proj_{}'.format(i + 1))
        conv_proj = conv1d(conv_proj, 3, c[-1], None, is_training, 'proj_{}'.format(len(c)))

        # Residual connection:
        highway_input = conv_proj + inputs

        # Handle dimensionality mismatch:
        if highway_input.shape[2] != 128:
            highway_input = tf.layers.dense(highway_input, 128)

        # 4-layer HighwayNet:
        h = highway_input
        for i in range(num_highways):
            with tf.variable_scope('highway_' + str(i)):
                # site specific speaker embedding
                if speaker_embd is not None:
                    s = tf.layers.dense(speaker_embd, h.shape[-1], activation=tf.nn.softsign)
                    s = tf.tile(tf.expand_dims(s, 1), [1, tf.shape(h)[1], 1])
                    h = tf.concat([h, s], -1)
                h = highwaynet(h)

        # site specific speaker embedding
        if speaker_embd is not None:
            # TODO: what about two different s1, s2 for forwards and backwards
            s = tf.layers.dense(speaker_embd, gru_units, activation=tf.nn.softsign)
        else:
            s = None

        # Bidirectional RNN
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            GRUCell(gru_units),
            GRUCell(gru_units),
            h,
            initial_state_fw=s,
            initial_state_bw=s,
            sequence_length=input_lengths,
            dtype=tf.float32)
        encoded = tf.concat(outputs, axis=2)  # Concat forward and backward

        return encoded


def highwaynet(inputs, activation=tf.nn.relu, scope="highway", reuse=False):
    num_units = inputs.shape[-1]
    with tf.variable_scope(scope, reuse=reuse):
        h = tf.layers.dense(inputs, units=num_units, activation=activation, name='H')
        t = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid, name='T',
                            bias_initializer=tf.constant_initializer(-1.0))
        return h * t + inputs * (1.0 - t)


def conv1d(inputs, kernel_size, channels, activation, is_training, scope):
    with tf.variable_scope(scope):
        conv1d_output = tf.layers.conv1d(inputs, filters=channels, kernel_size=kernel_size,
                                         activation=activation, padding='same')
        return tf.layers.batch_normalization(conv1d_output, training=is_training)
