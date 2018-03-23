import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import librosa
import numpy as np

from neural_speech.utils import audio



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
    wav = trim_wav(audio.load_wav(wav_path))
    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav)
    n_frames = spectrogram.shape[1]
    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav)

    # Return a tuple describing this training example:
    return idx, wav, spectrogram.T, mel_spectrogram.T, n_frames


def trim_wav(wav, threshold_db=25):
    '''Trims silence from the ends of the wav'''
    splits = librosa.effects.split(wav, threshold_db, frame_length=1024, hop_length=512)
    return wav[_find_start(splits):_find_end(splits, len(wav))]


def trim_silence(wav, threshold, frame_length=2048):
    '''Removes silence at the beginning and end of a sample.'''
    if wav.size < frame_length:
        frame_length = wav.size
    energy = librosa.feature.rmse(wav, frame_length=frame_length)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return wav[indices[0]:indices[-1]] if indices.size else wav[:0]


def _find_start(splits, min_samples=2000):
    for split_start, split_end in splits:
        if split_end - split_start > min_samples:
            return max(0, split_start - min_samples)
    return 0


def _find_end(splits, num_samples, min_samples=2000):
    for split_start, split_end in reversed(splits):
        if split_end - split_start > min_samples:
            return min(num_samples, split_end + min_samples)
    return num_samples
