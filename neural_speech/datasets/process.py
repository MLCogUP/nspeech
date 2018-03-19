import os
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import librosa

from neural_speech.util import audio

_min_samples = 2000
_threshold_db = 25
_speaker_re = re.compile(r'p([0-9]+)_')


def build_from_path(filenames, out_dir, num_workers=1, tqdm=lambda x: x, limit=0):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    for wav_path, text, speaker_id, dataset_id in filenames:
        if limit and len(futures) > limit:
            break
        futures.append(executor.submit(partial(process_utterance, wav_path, dataset_id)))
    return [future.result() for future in tqdm(futures)]


# TODO cached works worse than non-cached?!? omg! what's wrong?
# TODO work on h5py data for caching
def process_utterance(wav_path, dataset_id):
    idx = os.path.basename(wav_path)[:-4]

    # Load the audio to a numpy array:
    # wav = _trim_wav(audio.load_wav(wav_path))
    wav = _trim_wav(audio.load_wav(wav_path))
    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav)
    n_frames = spectrogram.shape[1]
    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav)

    # Return a tuple describing this training example:
    return idx, wav, spectrogram.T, mel_spectrogram.T, n_frames


def _trim_wav(wav):
    '''Trims silence from the ends of the wav'''
    splits = librosa.effects.split(wav, _threshold_db, frame_length=1024, hop_length=512)
    return wav[_find_start(splits):_find_end(splits, len(wav))]


def _find_start(splits):
    for split_start, split_end in splits:
        if split_end - split_start > _min_samples:
            return max(0, split_start - _min_samples)
    return 0


def _find_end(splits, num_samples):
    for split_start, split_end in reversed(splits):
        if split_end - split_start > _min_samples:
            return min(num_samples, split_end + _min_samples)
    return num_samples
