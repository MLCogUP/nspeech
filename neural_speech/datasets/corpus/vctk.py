import glob
import os
import re


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
                yield wav_path, text, speaker_id, "vctk"
