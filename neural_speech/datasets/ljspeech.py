import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

from util import audio


def load_file_names(in_dir):
    with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
            text = parts[2]
            speaker_id = 0
            yield wav_path, text, speaker_id


def load_libre_2(in_dir):
    # example (line break over comma)
    # 1272-128104-0012,
    # dev-clean/dev-clean/1272/128104/1272-128104-0012.flac,
    # only unfortunately his own work never does get good,
    # training
    with open(os.path.join(in_dir, 'corpus.csv'), encoding='utf-8') as f:
        for line in f:
            identifier, path, text, mode = line.strip().split(',')
            speaker_id, chapter, utterance = identifier.split("-")
            wav_path = os.path.join(in_dir, path)
            # TODO: need filters? e.g. mode is training
            yield wav_path, text, speaker_id


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    '''Preprocesses the LJ Speech dataset from a given input path into a given output directory.

      Args:
        in_dir: The directory where you have downloaded the LJ Speech dataset
        out_dir: The directory to write the output into
        num_workers: Optional number of worker processes to parallelize across
        tqdm: You can optionally pass tqdm to get a nice progress bar

      Returns:
        A list of tuples describing the training examples. This should be written to train.txt
    '''
    # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
    # can omit it and just call _process_utterance on each input if you want.
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1
    with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
            text = parts[2]
            futures.append(executor.submit(partial(_process_utterance, out_dir, index, wav_path, text)))
            # futures.append(_process_utterance(out_dir, index, wav_path, text))
            index += 1
    return [future.result() for future in tqdm(futures)]
    # return futures


def _process_utterance(out_dir, index, wav_path, text):
    '''Preprocesses a single utterance audio/text pair.

    This writes the mel and linear scale spectrograms to disk and returns a tuple to write
    to the train.txt file.

    Args:
      out_dir: The directory to write the spectrograms into
      index: The numeric index to use in the spectrogram filenames.
      wav_path: Path to the audio file containing the speech input
      text: The text spoken in the input audio file

    Returns:
      A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
    '''
    spectrogram_fn = 'ljspeech-spec-%05d.npy' % index
    spectrogram_path = os.path.join(out_dir, spectrogram_fn)
    mel_fn = 'ljspeech-mel-%05d.npy' % index
    mel_path = os.path.join(out_dir, mel_fn)
    wav_fn = os.path.basename(wav_path)

    if os.path.exists(spectrogram_path):
        n_frames = np.load(spectrogram_path).shape[1]
    else:
        # Load the audio to a numpy array:
        wav = audio.load_wav(wav_path)

        # Compute the linear-scale spectrogram from the wav:
        spectrogram = audio.spectrogram(wav).astype(np.float32)
        n_frames = spectrogram.shape[1]

        # Write the spectrograms to disk:
        np.save(spectrogram_path, spectrogram.T, allow_pickle=False)

        if not os.path.exists(mel_path):
            # Compute a mel-scale spectrogram from the wav:
            mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

            # Write the spectrograms to disk:
            np.save(mel_path, mel_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example:
    return wav_fn, spectrogram_fn, mel_fn, n_frames, text, 1
