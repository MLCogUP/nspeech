import models.rnn_wrappers
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell


def embedding(inputs, vocab_size, num_units, scope="embedding", reuse=None):
    '''
    Embeds a given tensor.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        embedding_table = tf.get_variable('embedding',
                                          [vocab_size, num_units],
                                          dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                      stddev=0.01))  # stddev=0.5?
    return tf.nn.embedding_lookup(embedding_table, inputs)  # [N, T_in, 256]


def prenet(inputs, drop_rate, is_training, layer_sizes, scope="prenet", reuse=None):
    x = inputs
    with tf.variable_scope(scope, reuse=reuse):
        for i, size in enumerate(layer_sizes):
            dense = tf.layers.dense(x, units=size, activation=tf.nn.elu, name='dense_%d' % (i + 1))
            x = tf.layers.dropout(dense, rate=drop_rate, name='dropout_%d' % (i + 1))
    return x


def attention_decoder(inputs, num_units, input_lengths, is_training, attention_type="bah",
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
        else:
            raise Exception("Unknown attention type ")

        # Attention
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                models.rnn_wrappers.PrenetWrapper(GRUCell(num_units), [256, 128], is_training),  # 256
                attention_mechanism,  # 256
                alignment_history=True,
                output_attention=False)  # [N, T_in, 256]
        #  Concatenate attention context vector and RNN cell output into a 512D vector.
        concat_cell = models.rnn_wrappers.ConcatOutputAndAttentionWrapper(attention_cell)  # [N, T_in, 512]
    return concat_cell


def cbhg(inputs, input_lengths, speaker_embed=None, is_training=True,
         K=16, c=(128, 128), gru_units=128, num_highways=4, scope="cbhg"):
    with tf.variable_scope(scope):
        with tf.variable_scope('conv_bank'):
            # Convolution bank: concatenate on the last axis to stack channels from all convolutions
            conv_outputs = tf.concat(
                    [conv1d(inputs, k, 128, tf.nn.elu, is_training, 'conv1d_%d' % k) for k in range(1, K + 1)],
                    axis=-1
            )

        # Maxpooling:
        conv_bank = tf.layers.max_pooling1d(conv_outputs, pool_size=2, strides=1, padding='same')

        # Two projection layers:
        conv_proj = conv1d(conv_bank, 3, c[0], tf.nn.elu, is_training, 'proj_1')
        conv_proj = conv1d(conv_proj, 3, c[1], None, is_training, 'proj_2')

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
                if speaker_embed is not None:
                    s = tf.layers.dense(speaker_embed, h.shape[-1], activation=tf.nn.elu)
                    s = tf.tile(tf.expand_dims(s, 1), [1, tf.shape(h)[1], 1])
                    h = tf.concat([h, s], 2)
                h = highwaynet(h)

        # site specfic speaker embedding
        if speaker_embed is not None:
            s = tf.layers.dense(speaker_embed, gru_units, activation=tf.nn.elu)
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


def highwaynet(inputs, num_units=128, scope="highway", reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        h = tf.layers.dense(inputs, units=num_units, activation=tf.nn.elu, name='H')
        t = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid, name='T',
                            bias_initializer=tf.constant_initializer(-1.0))
        return h * t + inputs * (1.0 - t)


def conv1d(inputs, kernel_size, channels, activation, is_training, scope):
    with tf.variable_scope(scope):
        conv1d_output = tf.layers.conv1d(inputs, filters=channels, kernel_size=kernel_size,
                                         activation=activation, padding='same')
        return tf.layers.batch_normalization(conv1d_output, training=is_training)
