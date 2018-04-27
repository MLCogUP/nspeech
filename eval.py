import argparse
import os
import re

import neural_speech.hparams
from neural_speech.synthesizer import Synthesizer
from neural_speech.utils import plot, audio

sentences = [
    # From July 8, 2017 New York Times:
    'Scientists at the CERN laboratory say they have discovered a new particle.',
    'There’s a way to measure the acute emotional intelligence that has never gone out of style.',
    'President Trump met with other leaders at the Group of 20 conference.',
    'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
    # From Google's Tacotron example page:
    'Generative adversarial network or variational auto-encoder.',
    'The buses aren\'t the problem, they actually provide a solution.',
    'Does the quick brown fox jump over the lazy dog?',
    'Talib Kweli confirmed to AllHipHop that he will be releasing an album in the next year.',
]


def get_output_base_path(checkpoint_path):
    base_dir = os.path.dirname(checkpoint_path)
    m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
    name = 'eval-%d' % int(m.group(1)) if m else 'eval'
    return os.path.join(base_dir, name)


def run_eval(args, hparams):
    synth = Synthesizer(hparams)
    synth.load(args.checkpoint, args.model)
    base_path = get_output_base_path(args.checkpoint)
    simple_eval(args, synth, base_path)
    harvard_eval(args, synth, base_path)


def simple_eval(args, synth, base_path):
    for i, text in enumerate(sentences):
        path = '%s-%d' % (base_path, i)
        print('Synthesizing: %s' % path)
        wav, mel, lin = synth.synthesize(text, args.speaker)
        audio.save_wav(wav, path + ".wav")
        plot.plot_specgram(lin, path + "-lin.png", "linear")
        plot.plot_specgram(mel, path + "-mel.png", "mel")


def harvard_eval(args, synth, base_path):
    sentences = open('harvard_sentences.txt', 'r').readlines()
    for i, text in enumerate(sentences):
        if i % 11 == 0: continue
        if i / 11 > 3: break
        text = " ".join(text.split()[1:])
        path = '%s-%d-%d.wav' % (base_path, int(i / 11), i % 11)
        print('Synthesizing: %s' % path)
        wav, mel, lin = synth.synthesize(text, args.speaker)
        audio.save_wav(wav, path + ".wav")
        plot.plot_specgram(lin, path + "-lin.png", "linear")
        plot.plot_specgram(mel, path + "-mel.png", "mel")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--model', default='tacotron')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--gpu', default=0, type=int, help='Select gpu for computation')
    parser.add_argument('--speaker', type=int, default=-1, help='Speaker ID')
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)  # added available gpu
    hparams = neural_speech.hparams.load(args.model)
    hparams.parse(args.hparams)
    neural_speech.hparams.debug_string(hparams)
    run_eval(args, hparams)


if __name__ == '__main__':
    main()
