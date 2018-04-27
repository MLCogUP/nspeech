import numpy as np
import tensorflow as tf

from neural_speech.models import create_model
from neural_speech.utils import audio
from neural_speech.utils.text import text_to_sequence


class Synthesizer:
    def __init__(self, hparams):
        self.model = None
        self.wav_output = None
        self.mel_out = None
        self.lin_out = None
        self.session = None
        self.hparams = hparams

    def load(self, checkpoint_path, model_name):
        print('Constructing model: %s' % model_name)
        inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
        input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
        if self.hparams.num_speakers > 1:
            speaker_ids = tf.placeholder(tf.int32, [1], 'speaker_ids')
        else:
            speaker_ids = None
        with tf.variable_scope('model'):
            self.model = create_model(model_name, self.hparams)
            self.model.initialize(inputs, input_lengths, speaker_ids)
            # TODO: add more outputs as spectrograms
            self.wav_output = audio.inv_spectrogram_tensorflow(self.model.linear_outputs[0])
            self.mel_out = self.model.mel_outputs[0]
            self.lin_out = self.model.linear_outputs[0]

        print('Loading checkpoint: %s' % checkpoint_path)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_path)

    def synthesize(self, text, speaker_id):
        cleaner_names = [x.strip() for x in self.hparams.cleaners.split(',')]
        seq = text_to_sequence(text, cleaner_names)
        print("text:", text)
        print("input:", seq)
        feed_dict = {
            self.model.inputs: [np.asarray(seq, dtype=np.int32)],
            self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
        }
        if speaker_id >= 0 and self.hparams.num_speakers > 1:
            feed_dict[self.model.speaker_ids] = np.asarray([speaker_id], dtype=np.int32)
        wav, mel, lin = self.session.run([self.wav_output, self.mel_out, self.lin_out], feed_dict=feed_dict)
        wav = audio.inv_preemphasis(wav)
        wav = wav[:audio.find_endpoint(wav)]
        return wav, mel, lin
