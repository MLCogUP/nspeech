import random
import threading
import time
import traceback

import matplotlib
import numpy as np
import tensorflow as tf

import datasets.corpus.ljspeech
import datasets.corpus.vctk
import datasets.corpus.vctk
from datasets.process import _process_utterance
from text import text_to_sequence
from util.infolog import log

matplotlib.use('Agg')

_p_cmudict = 0.5
_pad = 0


# TODO: maybe update to use tf thread handling?
# https://www.tensorflow.org/api_guides/python/threading_and_queues#Queue_usage_overview
class DataFeeder(object):
    '''Feeds batches of data into a queue on a background thread.'''

    def __init__(self, sess, coordinator, input_paths, hparams):
        self._coord = coordinator
        self._hparams = hparams
        self._cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
        self._offset = 0
        self._session = sess
        self._threads = []

        self._data_items = []

        self.load_data(input_paths)
        self.add_placeholders()
        self.add_cmudict()

    def load_data(self, input_paths):
        # TODO: support more corpora by this function
        path_to_function = {
            "vctk": datasets.corpus.vctk.load_file_names,
            "ljspeech": datasets.corpus.ljspeech.load_file_names,
            "librispeech": datasets.corpus.ljspeech.load_libre_2
        }
        for data_type, data_source in input_paths.items():
            self._data_items.extend(list(path_to_function[data_type](data_source)))

        log('Loaded data refs for %d examples' % len(self._data_items))
        assert len(self._data_items) > 0, "No data found"

        _, self.speaker_cardinality = get_category_cardinality(self._data_items)
        self.speaker_cardinality += 1

    def add_placeholders(self):
        hp = self._hparams
        # Create placeholders for inputs and targets. Don't specify batch size because we want to
        # be able to feed different sized batches at eval time.
        self._placeholders = [
            tf.placeholder(tf.int32, [None, None], 'inputs'),
            tf.placeholder(tf.int32, [None], 'input_lengths'),
            tf.placeholder(tf.int32, [None], 'speaker_ids'),
            tf.placeholder(tf.float32, [None, None, hp.num_mels], 'mel_targets'),
            tf.placeholder(tf.float32, [None, None, hp.num_freq], 'linear_targets'),
            tf.placeholder(tf.float32, [None, None], 'audio')
        ]

        # Create queue for buffering data:
        queue = tf.RandomShuffleQueue(hp.queue_size,
                                      min_after_dequeue=int(hp.min_dequeue_ratio * hp.queue_size),
                                      dtypes=[tf.int32, tf.int32, tf.int32, tf.float32, tf.float32, tf.float32],
                                      name='input_queue')
        self._enqueue_op = queue.enqueue(self._placeholders)
        self.inputs, self.input_lengths, self.speaker_ids, self.mel_targets, self.linear_targets, self.audio = queue.dequeue()
        self.inputs.set_shape(self._placeholders[0].shape)
        self.input_lengths.set_shape(self._placeholders[1].shape)
        self.speaker_ids.set_shape(self._placeholders[2].shape)
        self.mel_targets.set_shape(self._placeholders[3].shape)
        self.linear_targets.set_shape(self._placeholders[4].shape)
        self.audio.set_shape(self._placeholders[5].shape)

    def add_cmudict(self):
        # Load CMUDict: If enabled, this will randomly substitute some words in the training data with
        # their ARPABet equivalents, which will allow you to also pass ARPABet to the model for
        # synthesis (useful for proper nouns, etc.)
        # if hparams.use_cmudict:
        #     cmudict_path = os.path.join(self._datadir, 'cmudict-0.7b')
        #     if not os.path.isfile(cmudict_path):
        #         raise Exception('If use_cmudict=True, you must download ' +
        #                         'http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b to %s' % cmudict_path)
        #     self._cmudict = cmudict.CMUDict(cmudict_path, keep_ambiguous=False)
        #     log('Loaded CMUDict with %d unambiguous entries' % len(self._cmudict))
        # else:
        self._cmudict = None

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
        Loads a bunch of samples into memory, sorts them by output length and
        creates batches from it.
        This is done for efficiency - reducing padding.
        """
        start = time.time()

        # Read a group of examples:
        n = self._hparams.batch_size
        r = self._hparams.outputs_per_step
        batches_per_group = self._hparams.batch_group_size
        examples = [self._get_next_example() for _ in range(n * batches_per_group)]

        # Bucket examples based on similar output sequence length for efficiency:
        examples.sort(key=lambda x: x[-1])
        # TODO: numpy split array?
        batches = [examples[i:i + n] for i in range(0, len(examples), n)]
        random.shuffle(batches)

        log('Generated %d batches of size %d in %.03f sec' % (len(batches), n, time.time() - start))
        for batch in batches:
            feed_dict = dict(zip(self._placeholders, _prepare_batch(batch, r)))
            self._session.run(self._enqueue_op, feed_dict=feed_dict)

    def _get_next_example(self):
        '''Loads a single example (input, speaker_id, mel_target, linear_target, cost) from disk'''
        if self._offset >= len(self._data_items):
            self._offset = 0
            random.shuffle(self._data_items)
        wav_path, text, speaker_id = self._data_items[self._offset]
        self._offset += 1

        wav_fn, wav, linear_target, mel_target, n_frames = _process_utterance(wav_path)

        if self._cmudict and random.random() < _p_cmudict:
            text = ' '.join([self._maybe_get_arpabet(word) for word in text.split(' ')])
        # encode text sequence information
        input_data = np.asarray(text_to_sequence(text, self._cleaner_names), dtype=np.int32)
        return input_data, wav, speaker_id, mel_target, linear_target, len(linear_target)

    def _maybe_get_arpabet(self, word):
        arpabet = self._cmudict.lookup(word)
        return '{%s}' % arpabet[0] if arpabet is not None and random.random() < 0.5 else word


def get_category_cardinality(files):
    ids = list(map(lambda x: int(x[2]), files))
    return min(ids), max(ids)


def _prepare_batch(batch, outputs_per_step):
    random.shuffle(batch)
    inputs = _prepare_inputs([x[0] for x in batch])
    # print(">>", inputs.shape)
    input_lengths = np.asarray([len(x[0]) for x in batch], dtype=np.int32)
    # print(">>", batch[0][1].shape)
    audios = _prepare_inputs([x[1] for x in batch])
    # print(">>", audios.shape)
    speaker_ids = np.asarray([x[2] for x in batch], dtype=np.int32)
    mel_targets = _prepare_targets([x[3] for x in batch], outputs_per_step)
    linear_targets = _prepare_targets([x[4] for x in batch], outputs_per_step)
    return inputs, input_lengths, speaker_ids, mel_targets, linear_targets, audios


def _prepare_inputs(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_input(x, max_len) for x in inputs])


def _prepare_targets(targets, alignment):
    max_len = max((len(t) for t in targets)) + 1
    return np.stack([_pad_target(t, _round_up(max_len, alignment)) for t in targets])


def _pad_input(x, length):
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def _pad_target(t, length):
    return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=_pad)


def _round_up(x, multiple):
    remainder = x % multiple
    return x if remainder == 0 else x + multiple - remainder
