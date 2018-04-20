import os
import random
import threading
import traceback

import joblib
import numpy as np
import tensorflow as tf
from scipy.misc import imresize

import neural_speech.datasets.corpus
import neural_speech.datasets.process
from neural_speech.utils import log, audio
from neural_speech.utils.text import text_to_sequence


class WavenetDataFeeder(object):
    '''Feeds batches of data into a queue on a background thread.'''

    def __init__(self, sess, coordinator, input_paths, receptive_field, hparams):
        self._coord = coordinator
        self._hparams = hparams
        self._cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
        self._offset = 0
        self._session = sess
        self._threads = []

        self.silence_threshold = 0.1
        self.sample_size = hparams.sample_size
        self.receptive_field = receptive_field

        self._data_items = []

        self.load_data(input_paths)
        self.add_placeholders()

    def load_data(self, input_paths):
        # TODO: support more corpora by this function
        path_to_function = {
            "vctk": neural_speech.datasets.corpus.vctk.load_file_names,
            "ljspeech": neural_speech.datasets.corpus.ljspeech.load_file_names,
            "librispeech": neural_speech.datasets.corpus.ljspeech.load_libre_2
        }
        for data_type, data_source in input_paths.items():
            self._data_items.extend(list(path_to_function[data_type](data_source)))

        if os.path.isfile("/cache/id2speaker.joblib"):
            self.id2speaker = joblib.load("/cache/id2speaker.joblib")
        speakers = {(dataset_id, speaker_id) for (_, _, speaker_id, dataset_id) in self._data_items}
        self.id2speaker.update(dict(enumerate(speakers)))
        joblib.dump(self.id2speaker, "/cache/id2speaker.joblib")
        self.speaker2id = {v: k for k, v in self.id2speaker.items()}

        log('Loaded data refs for %d examples' % len(self._data_items))
        log('Loaded %d different speaker(s)' % len(self.speaker2id))
        assert len(self._data_items) > 0, "No data found"

    def add_placeholders(self):
        hp = self._hparams
        # Create placeholders for inputs and targets. Don't specify batch size because we want to
        # be able to feed different sized batches at eval time.
        audio_length = self.receptive_field + self.sample_size
        self._placeholders = [
            tf.placeholder(tf.float32, [audio_length], 'audio'),
            tf.placeholder(tf.int32, [1], 'speaker_ids'),
            tf.placeholder(tf.float32, [audio_length - self.sample_size, hp.num_freq], 'linear_targets'),
            tf.placeholder(tf.float32, [audio_length - self.sample_size, hp.num_mels], 'mel_targets'),
        ]

        # Create queue for buffering data:
        queue = tf.RandomShuffleQueue(hp.queue_size,
                                      min_after_dequeue=int(hp.min_dequeue_ratio * hp.queue_size),
                                      dtypes=[tf.float32, tf.int32, tf.float32, tf.float32],
                                      shapes=[[audio_length],
                                              [1],
                                              [audio_length - self.sample_size, hp.num_freq],
                                              [audio_length - self.sample_size, hp.num_mels]],
                                      name='input_queue')
        self.size = queue.size()
        self.capacity = hp.queue_size
        self._enqueue_op = queue.enqueue(self._placeholders)
        self.audio, self.speaker_ids, self.linear_targets, self.mel_targets = queue.dequeue_many(hp.batch_size)

    def start_threads(self, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=())
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self._threads.append(thread)
        return self._threads

    def thread_main(self):
        try:
            stop = False
            # Go through the dataset multiple times
            while not stop:
                self._enqueue_next_group()
                if self._coord.should_stop():
                    stop = True
        except Exception as e:
            traceback.print_exc()
            self._coord.request_stop(e)

    def _enqueue_next_group(self):
        """
        Loads one wav example at a time and creates several training samples from it based on the receptive field
        and the number of predictions per sample.
        """
        # Read a group of examples:
        wav_fn, wav, text, speaker_id = self._get_next_example()

        # trim wav
        if self.silence_threshold is not None:
            # Remove silence
            wav = neural_speech.datasets.process.trim_silence(wav, self.silence_threshold)
            if wav.size == 0:
                print("Warning: {} was ignored as it contains only "
                      "silence. Consider decreasing trim_silence "
                      "threshold, or adjust volume of the audio.".format(wav_fn))

        wav = np.pad(wav, [self.receptive_field, 0], 'constant')

        # Cut samples into pieces of size receptive_field +
        # sample_size with receptive_field overlap
        while len(wav) > self.receptive_field + self.sample_size:
            piece = wav[:(self.receptive_field + self.sample_size)]
            self.enqueue_audio(piece, speaker_id)
            wav = wav[self.sample_size:]

    def enqueue_audio(self, wav, speaker_id):
        linear = audio.spectrogram(wav).T
        mel = audio.melspectrogram(wav).T

        linear = imresize(linear, size=(len(wav) - self.sample_size, linear.shape[1]))
        mel = imresize(mel, size=(len(wav) - self.sample_size, mel.shape[1]))

        feed_dict = dict(zip(self._placeholders, (wav, speaker_id, linear, mel)))
        self._session.run(self._enqueue_op, feed_dict=feed_dict)

    def _get_next_example(self):
        '''Loads a single example (input, speaker_id, mel_target, linear_target, cost) from disk'''
        if self._offset >= len(self._data_items):
            self._offset = 0
            random.shuffle(self._data_items)
        wav_path, text, local_speaker_id, dataset_id = self._data_items[self._offset]
        speaker_id = self.speaker2id[dataset_id, local_speaker_id]

        self._offset += 1

        # Load the audio to a numpy array:
        wav_fn = os.path.basename(wav_path)
        wav = audio.load_wav(wav_path)

        # encode text sequence information
        text = np.asarray(text_to_sequence(text, self._cleaner_names), dtype=np.int32)
        return wav_fn, wav, text, np.asarray([speaker_id], dtype=np.int32)
