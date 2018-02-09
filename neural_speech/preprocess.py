# -*- coding: utf-8 -*-

import argparse
import multiprocessing
import os

from tqdm import tqdm

import datasets
from hparams import hparams


def preprocess_blizzard(args, in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    metadata = datasets.corpus.blizzard.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)


def preprocess_ljspeech(args, in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    metadata = datasets.corpus.ljspeech.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)


def preprocess_german_speech(args, in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    metadata = datasets.corpus.german_speech.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)


def preprocess_pavoque(args, in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    metadata = datasets.corpus.pavoque_corpus.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)


def preprocess_vctk(args, in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    metadata = datasets.corpus.vctk.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm, limit=args.limit)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    """
    assumes metadata to have 5 tuples of data: wav_fn, linspec, melspec, duration, text
    """
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    frames = sum([meta[3] for meta in metadata])  # select nframes from meta data
    hours = frames * hparams.frame_shift_ms / (3600 * 1000)
    print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))
    print('Max input length:  %d' % max(len(text) for (wav_fn, linspec, melspec, duration, text, _) in metadata))
    print('Max output length: %d' % max(duration for (wav_fn, linspec, melspec, duration, text, _) in metadata))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', default="data")
    parser.add_argument('--out_dir', default='training')
    parser.add_argument('--dataset', required=True,
                        choices=['blizzard', 'ljspeech', 'german_speech', 'pavoque', 'vctk'])
    parser.add_argument('--limit', default=0)
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()
    if args.dataset == 'blizzard':
        preprocess_blizzard(args, args.in_dir, args.out_dir)
    elif args.dataset == 'ljspeech':
        preprocess_ljspeech(args, args.in_dir, args.out_dir)
    elif args.dataset == 'german_speech':
        preprocess_german_speech(args, args.in_dir, args.out_dir)
    elif args.dataset == 'pavoque':
        preprocess_pavoque(args, args.in_dir, args.out_dir)
    elif args.dataset == 'vctk':
        preprocess_vctk(args, args.in_dir, args.out_dir)

if __name__ == "__main__":
    main()
