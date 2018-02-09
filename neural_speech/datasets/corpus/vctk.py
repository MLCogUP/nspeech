import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import librosa
import numpy as np

from util import audio

_min_samples = 2000
_threshold_db = 25
_speaker_re = re.compile(r'p([0-9]+)_')


def load_file_names(in_dir):
    wav_paths = glob.glob('%s/wav48/p*/*.wav' % in_dir)
    for wav_path in wav_paths:
        text_path = wav_path.replace('wav48', 'txt').replace('wav', 'txt')
        if os.path.isfile(text_path):
            with open(text_path, 'r') as f:
                text = f.read().strip()
                name = os.path.splitext(os.path.basename(wav_path))[0]
                speaker_id = _speaker_re.match(name).group(1)
                yield wav_path, text, speaker_id


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x, limit=0):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    for wav_path, text, speaker_id in load_file_names(in_dir):
        if limit and len(futures) > limit:
            break
        futures.append(executor.submit(partial(_process_utterance, out_dir, wav_path, text, speaker_id)))
    return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, wav_path, text, speaker_id):
    wav_fn = os.path.basename(wav_path)
    name = os.path.splitext(os.path.basename(wav_path))[0]
    spectrogram_fn = 'vctk-linear-%s.npy' % name
    spectrogram_path = os.path.join(out_dir, spectrogram_fn)
    mel_fn = 'vctk-mel-%s.npy' % name
    mel_path = os.path.join(out_dir, mel_fn)

    if os.path.exists(spectrogram_path):
        n_frames = np.load(spectrogram_path).shape[1]
    else:
        # Load the audio to a numpy array:
        wav = _trim_wav(audio.load_wav(wav_path))

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
    return wav_fn, spectrogram_fn, mel_fn, n_frames, text, speaker_id


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
