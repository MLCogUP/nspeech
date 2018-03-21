import os


def load_file_names(in_dir):
    with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
            text = parts[2]
            speaker_id = 0
            yield wav_path, text, speaker_id, "ljspeech"


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
            yield wav_path, text, speaker_id, "libre"
