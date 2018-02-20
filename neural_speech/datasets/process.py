import os

import librosa
import numpy as np

from util import audio


def _process_utterance(wav_path):
    wav_fn = os.path.basename(wav_path)

    # Load the audio to a numpy array:
    # wav = _trim_wav(audio.load_wav(wav_path))
    wav = trim_wav(audio.load_wav(wav_path))

    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav)
    n_frames = spectrogram.shape[1]

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav)

    # Return a tuple describing this training example:
    return wav_fn, wav, spectrogram.T, mel_spectrogram.T, n_frames


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
