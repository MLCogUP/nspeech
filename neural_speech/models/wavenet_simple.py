import sys

sys.path.append("..")

import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use('Agg')


def create_variable(name, shape):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable


def create_embedding_table(name, shape):
    if shape[0] == shape[1]:
        # Make a one-hot encoding as the initial value.
        initial_val = np.identity(n=shape[0], dtype=np.float32)
        return tf.Variable(initial_val, name=name)
    else:
        return create_variable(name, shape)


def create_bias_variable(name, shape):
    '''Create a bias variable with the specified name and shape and initialize
    it to zero.'''
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)


class SimpleWaveNetModel(object):
    '''Implements the WaveNet network for generative audio.
    Usage (with the architecture as in the DeepMind paper):
        dilations = [2**i for i in range(N)] * M
        filter_width = 2  # Convolutions just use 2 samples.
        residual_channels = 16  # Not specified in the paper.
        dilation_channels = 32  # Not specified in the paper.
        skip_channels = 16      # Not specified in the paper.
        net = WaveNetModel(batch_size, dilations, filter_width,
                           residual_channels, dilation_channels,
                           skip_channels)
        loss = net.loss(input_batch)
    '''

    def __init__(self,
                 # batch_size,
                 # dilations,
                 # filter_width,
                 # residual_channels,
                 # dilation_channels,
                 # skip_channels,
                 # quantization_channels=2 ** 8,
                 # use_biases=False,
                 # scalar_input=False,
                 # initial_filter_width=32,
                 # histograms=False,
                 # global_condition_channels=None,
                 # global_condition_cardinality=None):
                 hparams):
        '''Initializes the WaveNet model.
        Args:
            batch_size: How many audio files are supplied per batch
                (recommended: 1).
            dilations: A list with the dilation factor for each layer.
            filter_width: The samples that are included in each convolution,
                after dilating.
            residual_channels: How many filters to learn for the residual.
            dilation_channels: How many filters to learn for the dilated
                convolution.
            skip_channels: How many filters to learn that contribute to the
                quantized softmax output.
            quantization_channels: How many amplitude values to use for audio
                quantization and the corresponding one-hot encoding.
                Default: 256 (8-bit quantization).
            use_biases: Whether to add a bias layer to each convolution.
                Default: False.
            scalar_input: Whether to use the quantized waveform directly as
                input to the network instead of one-hot encoding it.
                Default: False.
            initial_filter_width: The width of the initial filter of the
                convolution applied to the scalar input. This is only relevant
                if scalar_input=True.
            histograms: Whether to store histograms in the summary.
                Default: False.
            global_condition_channels: Number of channels in (embedding
                size) of global conditioning vector. None indicates there is
                no global conditioning.
            global_condition_cardinality: Number of mutually exclusive
                categories to be embedded in global condition embedding. If
                not None, then this implies that global_condition tensor
                specifies an integer selecting which of the N global condition
                categories, where N = global_condition_cardinality. If None,
                then the global_condition tensor is regarded as a vector which
                must have dimension global_condition_channels.
        '''
        self._hparams = hparams
        self.batch_size = hparams.batch_size
        self.dilations_length = hparams.dilations_length
        self.dilations_depth = hparams.dilations_depth
        self.dilations = [2 ** i for _ in range(self.dilations_depth) for i in range(self.dilations_length)]
        self.filter_width = hparams.filter_width
        self.residual_channels = hparams.residual_channels
        self.dilation_channels = hparams.dilation_channels
        self.quantization_channels = hparams.quantization_channels
        self.use_biases = hparams.use_biases
        self.skip_channels = hparams.skip_channels
        self.initial_filter_width = hparams.initial_filter_width
        self.global_condition_channels = hparams.gc_channels or None
        self.global_condition_cardinality = hparams.gc_category_cardinality or None
        self.local_condition_channels = hparams.lc_channels

        self.receptive_field = SimpleWaveNetModel.calculate_receptive_field(
            self.filter_width, self.dilations)
        self.variables = self._create_variables()

        # TODO initialize network

    # TODO: change to method?
    @staticmethod
    def calculate_receptive_field(filter_width, dilations):
        receptive_field = (filter_width - 1) * sum(dilations) + 1
        receptive_field += filter_width - 1
        return receptive_field

    def _create_variables(self):
        '''This function creates all variables used by the network.
        This allows us to share them between multiple calls to the loss
        function and generation function.'''

        var = dict()

        with tf.variable_scope('wavenet'):
            if self.global_condition_cardinality is not None:
                # We only look up the embedding if we are conditioning on a
                # set of mutually-exclusive categories. We can also condition
                # on an already-embedded dense vector, in which case it's
                # given to us and we don't need to do the embedding lookup.
                # Still another alternative is no global condition at all, in
                # which case we also don't do a tf.nn.embedding_lookup.
                with tf.variable_scope('embeddings'):
                    layer = dict()
                    layer['gc_embedding'] = create_embedding_table(
                        'gc_embedding',
                        [self.global_condition_cardinality,
                         self.global_condition_channels])
                    var['embeddings'] = layer

            with tf.variable_scope('causal_layer'):
                layer = dict()
                initial_channels = self.quantization_channels
                initial_filter_width = self.filter_width
                layer['filter'] = create_variable(
                    'filter',
                    [initial_filter_width,
                     initial_channels,
                     self.residual_channels])
                var['causal_layer'] = layer

            var['dilated_stack'] = list()
            with tf.variable_scope('dilated_stack'):
                for i, dilation in enumerate(self.dilations):
                    with tf.variable_scope('layer{}'.format(i)):
                        current = dict()
                        current['filter'] = create_variable(
                            'filter',
                            [self.filter_width,
                             self.residual_channels,
                             self.dilation_channels])
                        current['gate'] = create_variable(
                            'gate',
                            [self.filter_width,
                             self.residual_channels,
                             self.dilation_channels])
                        current['dense'] = create_variable(
                            'dense',
                            [1,
                             self.dilation_channels,
                             self.residual_channels])
                        current['skip'] = create_variable(
                            'skip',
                            [1,
                             self.dilation_channels,
                             self.skip_channels])

                        if self.global_condition_channels is not None:
                            current['gc_gateweights'] = create_variable(
                                'gc_gate',
                                [1, self.global_condition_channels,
                                 self.dilation_channels])
                            current['gc_filtweights'] = create_variable(
                                'gc_filter',
                                [1, self.global_condition_channels,
                                 self.dilation_channels])

                        # TODO: Add local condition batch
                        if self.local_condition_channels is not None:
                            current['lc_gateweights'] = create_variable(
                                'lc_gate',
                                [1, self.local_condition_channels,
                                 self.dilation_channels])
                            current['lc_filtweights'] = create_variable(
                                'lc_filter',
                                [1, self.local_condition_channels,
                                 self.dilation_channels])

                        if self.use_biases:
                            current['filter_bias'] = create_bias_variable(
                                'filter_bias',
                                [self.dilation_channels])
                            current['gate_bias'] = create_bias_variable(
                                'gate_bias',
                                [self.dilation_channels])
                            current['dense_bias'] = create_bias_variable(
                                'dense_bias',
                                [self.residual_channels])
                            current['skip_bias'] = create_bias_variable(
                                'slip_bias',
                                [self.skip_channels])

                        var['dilated_stack'].append(current)

            with tf.variable_scope('postprocessing'):
                current = dict()
                current['postprocess1'] = create_variable(
                    'postprocess1',
                    [1, self.skip_channels, self.skip_channels])
                current['postprocess2'] = create_variable(
                    'postprocess2',
                    [1, self.skip_channels, self.quantization_channels])
                if self.use_biases:
                    current['postprocess1_bias'] = create_bias_variable(
                        'postprocess1_bias',
                        [self.skip_channels])
                    current['postprocess2_bias'] = create_bias_variable(
                        'postprocess2_bias',
                        [self.quantization_channels])
                var['postprocessing'] = current

        return var

    def _create_causal_layer(self, input_batch):
        '''Creates a single causal convolution layer.
        The layer can change the number of channels.
        '''
        with tf.name_scope('causal_layer'):
            weights_filter = self.variables['causal_layer']['filter']
            return causal_conv(input_batch, weights_filter, 1)

    def _create_dilation_layer(self, input_batch, layer_index, dilation,
                               global_condition_batch, output_width, local_condition_batch):
        '''Creates a single causal dilated convolution layer.
        Args:
             input_batch: Input to the dilation layer.
             layer_index: Integer indicating which layer this is.
             dilation: Integer specifying the dilation size.
             global_conditioning_batch: Tensor containing the global data upon
                 which the output is to be conditioned upon. Shape:
                 [batch size, 1, channels]. The 1 is for the axis
                 corresponding to time so that the result is broadcast to
                 all time steps.
        The layer contains a gated filter that connects to dense output
        and to a skip connection:
               |-> [gate]   -|        |-> 1x1 conv -> skip output
               |             |-> (*) -|
        input -|-> [filter] -|        |-> 1x1 conv -|
               |                                    |-> (+) -> dense output
               |------------------------------------|
        Where `[gate]` and `[filter]` are causal convolutions with a
        non-linear activation at the output. Biases and global conditioning
        are omitted due to the limits of ASCII art.
        '''
        variables = self.variables['dilated_stack'][layer_index]

        weights_filter = variables['filter']
        weights_gate = variables['gate']
        conv_filter = causal_conv(input_batch, weights_filter, dilation)
        conv_gate = causal_conv(input_batch, weights_gate, dilation)

        if global_condition_batch is not None:
            weights_gc_filter = variables['gc_filtweights']
            weights_gc_gate = variables['gc_gateweights']
            conv_filter = conv_filter + tf.nn.conv1d(global_condition_batch,
                                                     weights_gc_filter,
                                                     stride=1,
                                                     padding="SAME",
                                                     name="gc_filter")
            conv_gate = conv_gate + tf.nn.conv1d(global_condition_batch,
                                                 weights_gc_gate,
                                                 stride=1,
                                                 padding="SAME",
                                                 name="gc_gate")

        # TODO: Add local condition batch
        if local_condition_batch is not None:
            weights_lc_filter = variables['lc_filtweights']
            weights_lc_gate = variables['lc_gateweights']
            conv_filter = conv_filter + tf.nn.conv1d(local_condition_batch,
                                                     weights_lc_filter,
                                                     stride=1,
                                                     padding="SAME",
                                                     name="lc_filter")
            conv_gate = conv_gate + tf.nn.conv1d(local_condition_batch,
                                                 weights_lc_gate,
                                                 stride=1,
                                                 padding="SAME",
                                                 name="lc_gate")

        if self.use_biases:
            filter_bias = variables['filter_bias']
            gate_bias = variables['gate_bias']
            conv_filter = tf.add(conv_filter, filter_bias)
            conv_gate = tf.add(conv_gate, gate_bias)

        out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

        # The 1x1 conv to produce the residual output
        weights_dense = variables['dense']
        transformed = tf.nn.conv1d(
            out, weights_dense, stride=1, padding="SAME", name="dense")

        # The 1x1 conv to produce the skip output
        skip_cut = tf.shape(out)[1] - output_width
        out_skip = tf.slice(out, [0, skip_cut, 0], [-1, -1, -1])
        weights_skip = variables['skip']
        skip_contribution = tf.nn.conv1d(
            out_skip, weights_skip, stride=1, padding="SAME", name="skip")

        if self.use_biases:
            dense_bias = variables['dense_bias']
            skip_bias = variables['skip_bias']
            transformed = transformed + dense_bias
            skip_contribution = skip_contribution + skip_bias

        input_cut = tf.shape(input_batch)[1] - tf.shape(transformed)[1]
        input_batch = tf.slice(input_batch, [0, input_cut, 0], [-1, -1, -1])

        return skip_contribution, input_batch + transformed

    def _create_network(self, input_batch, global_condition_batch, local_condition_batch):
        '''Construct the WaveNet network.'''
        outputs = []
        current_layer = input_batch

        # TODO add regular linear for initial skip in outputs as in magenta
        current_layer = self._create_causal_layer(current_layer)
        output_width = tf.shape(input_batch)[1] - self.receptive_field + 1

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    output, current_layer = self._create_dilation_layer(
                        current_layer, layer_index, dilation,
                        global_condition_batch, output_width, local_condition_batch)
                    outputs.append(output)

        with tf.name_scope('postprocessing'):
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            w1 = self.variables['postprocessing']['postprocess1']
            w2 = self.variables['postprocessing']['postprocess2']
            if self.use_biases:
                b1 = self.variables['postprocessing']['postprocess1_bias']
                b2 = self.variables['postprocessing']['postprocess2_bias']

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)
            transformed1 = tf.nn.relu(total)
            conv1 = tf.nn.conv1d(transformed1, w1, stride=1, padding="SAME")
            if self.use_biases:
                conv1 = tf.add(conv1, b1)
            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding="SAME")
            if self.use_biases:
                conv2 = tf.add(conv2, b2)

        return conv2

    def _one_hot(self, input_batch):
        '''One-hot encodes the waveform amplitudes.
        This allows the definition of the network as a categorical distribution
        over a finite set of possible amplitudes.
        '''
        with tf.name_scope('one_hot_encode'):
            encoded = tf.one_hot(
                input_batch,
                depth=self.quantization_channels,
                dtype=tf.float32)
            shape = [self.batch_size, -1, self.quantization_channels]
            encoded = tf.reshape(encoded, shape)
        return encoded

    def _embed_gc(self, global_condition):
        '''Returns embedding for global condition.
        :param global_condition: Either ID of global condition for
               tf.nn.embedding_lookup or actual embedding. The latter is
               experimental.
        :return: Embedding or None
        '''
        embedding = None
        if self.global_condition_cardinality is not None:
            # Only lookup the embedding if the global condition is presented
            # as an integer of mutually-exclusive categories ...
            embedding_table = self.variables['embeddings']['gc_embedding']
            embedding = tf.nn.embedding_lookup(embedding_table,
                                               global_condition)
        elif global_condition is not None:
            # ... else the global_condition (if any) is already provided
            # as an embedding.

            # In this case, the number of global_embedding channels must be
            # equal to the the last dimension of the global_condition tensor.
            gc_batch_rank = len(global_condition.get_shape())
            dims_match = (global_condition.get_shape()[gc_batch_rank - 1] ==
                          self.global_condition_channels)
            if not dims_match:
                raise ValueError('Shape of global_condition {} does not'
                                 ' match global_condition_channels {}.'.
                                 format(global_condition.get_shape(),
                                        self.global_condition_channels))
            embedding = global_condition

        if embedding is not None:
            embedding = tf.reshape(
                embedding,
                [self.batch_size, 1, self.global_condition_channels])

        return embedding

    def predict_proba(self, waveform, global_condition=None, name='wavenet'):
        '''Computes the probability distribution of the next sample based on
        all samples in the input waveform.
        If you want to generate audio by feeding the output of the network back
        as an input, see predict_proba_incremental for a faster alternative.'''
        with tf.name_scope(name):
            encoded = self._one_hot(waveform)
            gc_embedding = self._embed_gc(global_condition)
            raw_output = self._create_network(encoded, gc_embedding, local_condition_batch=None)
            out = tf.reshape(raw_output, [-1, self.quantization_channels])
            # Cast to float64 to avoid bug in TensorFlow
            proba = tf.cast(
                tf.nn.softmax(tf.cast(out, tf.float64)), tf.float32)
            last = tf.slice(
                proba,
                [tf.shape(proba)[0] - 1, 0],
                [1, self.quantization_channels])
            return tf.reshape(last, [-1])

    def initialize(self, audio_inputs, global_conditions=None, local_conditions=None):
        '''Creates a WaveNet network and returns the autoencoding loss.
        The variables are all scoped to the given name.
        '''
        with tf.name_scope("inference"):
            # add this for transformation
            input_batch = tf.reshape(audio_inputs, [self.batch_size, -1, 1])
            # We mu-law encode and quantize the input audioform.
            encoded_input = mu_law_encode(input_batch,
                                          self.quantization_channels)
            gc_embedding = self._embed_gc(global_conditions)
            encoded = self._one_hot(encoded_input)
            network_input = encoded
            # Cut off the last sample of network input to preserve causality.
            network_input_width = tf.shape(network_input)[1] - 1
            network_input = tf.slice(network_input, [0, 0, 0],
                                     [-1, network_input_width, -1])
            # network_input, target_output = tf.split(network_input, [network_input_width, 1], 1)

            raw_output = self._create_network(network_input, gc_embedding, local_condition_batch=local_conditions)

            self.encoded = encoded
            self.raw_output = raw_output

    def add_loss(self, l2_regularization_strength=None):
        with tf.name_scope('loss'):
            # Cut off the samples corresponding to the receptive field
            # for the first predicted sample.
            target_output = tf.slice(
                tf.reshape(
                    self.encoded,
                    [self.batch_size, -1, self.quantization_channels]),
                [0, self.receptive_field, 0],
                [-1, -1, -1])
            target_output = tf.reshape(target_output,
                                       [-1, self.quantization_channels])
            prediction = tf.reshape(self.raw_output,
                                    [-1, self.quantization_channels])
            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=prediction,
                labels=target_output)
            loss = tf.reduce_mean(loss)

            tf.summary.scalar('loss', loss)

            if l2_regularization_strength is not None:
                # L2 regularization for all trainable parameters
                l2_loss = tf.add_n([tf.nn.l2_loss(v)
                                    for v in tf.trainable_variables()
                                    if not ('bias' in v.name)])

                # Add the regularization term to the loss
                loss = (loss + l2_regularization_strength * l2_loss)

                tf.summary.scalar('l2_loss', l2_loss)
                tf.summary.scalar('total_loss', loss)
            self.loss = loss

    def add_optimizer(self, global_step, gradient_clip=1.0):
        '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.

        Args:
          global_step: int32 scalar Tensor representing current global step in training
        '''
        with tf.variable_scope('optimizer'):
            hp = self._hparams
            if hp.decay_learning_rate:
                self.learning_rate = _learning_rate_decay(hp.initial_learning_rate, global_step)
            else:
                self.learning_rate = tf.convert_to_tensor(hp.initial_learning_rate)
            optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.adam["beta1"], hp.adam["beta2"])
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
            # gradient_norms = [tf.norm(grad) for grad in self.gradients]
            # tf.summary.histogram('gradient_norm', gradient_norms)
            # tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))
            self.stats = tf.summary.merge_all()


def _learning_rate_decay(init_lr, global_step):
    # Noam scheme from tensor2tensor:
    warmup_steps = 4000.0
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


def causal_conv(value, filter_, dilation, name='causal_conv'):
    def time_to_batch(value, dilation, name="time_to_batch"):
        with tf.name_scope(name):
            shape = tf.shape(value)
            pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
            padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
            reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
            transposed = tf.transpose(reshaped, perm=[1, 0, 2])
            return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])

    def batch_to_time(value, dilation, name="batch_to_time"):
        with tf.name_scope(name):
            shape = tf.shape(value)
            prepared = tf.reshape(value, [dilation, -1, shape[2]])
            transposed = tf.transpose(prepared, perm=[1, 0, 2])
            return tf.reshape(transposed,
                              [tf.div(shape[0], dilation), -1, shape[2]])

    with tf.name_scope(name):
        filter_width = tf.shape(filter_)[0]
        if dilation > 1:
            transformed = time_to_batch(value, dilation)
            conv = tf.nn.conv1d(transformed, filter_, stride=1,
                                padding='VALID')
            restored = batch_to_time(conv, dilation)
        else:
            restored = tf.nn.conv1d(value, filter_, stride=1, padding='VALID')
        # Remove excess elements at the end.
        out_width = tf.shape(value)[1] - (filter_width - 1) * dilation
        result = tf.slice(restored,
                          [0, 0, 0],
                          [-1, out_width, -1])
        return result


def mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitudes.'''
    with tf.name_scope('encode'):
        mu = tf.to_float(quantization_channels - 1)
        # Perform mu-law companding transformation (ITU-T, 1988).
        # Minimum operation is here to deal with rare large amplitudes caused
        # by resampling.
        safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
        magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
        signal = tf.sign(audio) * magnitude
        # Quantize signal to the specified number of levels.
        return tf.to_int32((signal + 1) / 2 * mu + 0.5)


def mu_law_decode(output, quantization_channels):
    '''Recovers waveform from quantized values.'''
    with tf.name_scope('decode'):
        mu = quantization_channels - 1
        # Map values back to [-1, 1].
        signal = 2 * (tf.to_float(output) / mu) - 1
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu) ** abs(signal) - 1)
        return tf.sign(signal) * magnitude
