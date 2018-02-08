import matplotlib

matplotlib.use('Agg')

import argparse
import math
import os
import time
import tensorflow as tf
import traceback

from datasets.datafeeder import DataFeeder
from hparams import hparams, hparams_debug_string
from models import create_model
from text import sequence_to_text
from util import audio, ValueWindow, plot, get_git_commit, time_string, infolog
from util.infolog import log


def train(log_dir, args):
    commit = get_git_commit() if args.git else 'None'
    checkpoint_path = os.path.join(log_dir, 'model.ckpt')

    input_paths = {}
    if args.vctk:
        input_paths["vctk"] = os.path.join(args.base_dir, args.vctk)
    if args.ljspeech:
        input_paths["ljspeech"] = os.path.join(args.base_dir, args.ljspeech)

    log('Checkpoint path: %s' % checkpoint_path)
    log('Loading training data from: %s' % input_paths)
    log('Using model: %s' % args.model)
    log(hparams_debug_string())

    with tf.Session() as sess:
        # Set up DataFeeder:
        coord = tf.train.Coordinator()
        with tf.variable_scope('datafeeder'):
            feeder = DataFeeder(sess, coord, input_paths, hparams)

        # Set up model:
        global_step = tf.Variable(0, name='global_step', trainable=False)
        with tf.variable_scope('model'):
            model = create_model(args.model, hparams)
            model.initialize(feeder.inputs, feeder.input_lengths, feeder.speaker_ids, feeder.mel_targets,
                             feeder.linear_targets)
            model.add_loss()
            model.add_optimizer(global_step)
            model.add_stats()

        # Bookkeeping:
        time_window = ValueWindow(100)
        loss_window = ValueWindow(100)
        saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

        # Train!
        try:
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
            sess.run(tf.global_variables_initializer())

            if args.restore_step:
                # Restore from a checkpoint if the user requested it.
                restore_path = '%s-%d' % (checkpoint_path, args.restore_step)
                saver.restore(sess, restore_path)
                log('Resuming from checkpoint: %s at commit: %s' % (restore_path, commit), slack=True)
            else:
                log('Starting new training run at commit: %s' % commit, slack=True)

            feeder.start_threads()

            while not coord.should_stop():
                start_time = time.time()
                step, loss, opt = sess.run([global_step, model.loss, model.optimize])
                time_window.append(time.time() - start_time)
                loss_window.append(loss)
                message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f]' % (
                    step, time_window.average, loss, loss_window.average)
                log(message, slack=(step % args.checkpoint_interval == 0))

                if loss > 100 or math.isnan(loss):
                    log('Loss exploded to %.05f at step %d!' % (loss, step), slack=True)
                    raise Exception('Loss Exploded')

                if step % args.summary_interval == 0:
                    log('Writing summary at step: %d' % step)
                    summary_writer.add_summary(sess.run(model.stats), step)

                if step % args.checkpoint_interval == 0:
                    log('Saving checkpoint to: %s-%d' % (checkpoint_path, step))
                    saver.save(sess, checkpoint_path, global_step=step)
                    log('Saving audio and alignment...')
                    # compute one example
                    input_seq, spectrogram, alignment = sess.run([
                        model.inputs[0], model.linear_outputs[0], model.alignments[0]])
                    # TODO: replace with gpu griffinlim impl
                    waveform = audio.inv_spectrogram(spectrogram.T)
                    audio.save_wav(waveform, os.path.join(log_dir, 'step-%d-audio.wav' % step))
                    # np.save(os.path.join(log_dir, 'step-%d-align.npy' % step), alignment)
                    plot.plot_alignment(alignment, os.path.join(log_dir, 'step-%d-align.png' % step),
                                        info='%s, %s, %s, step=%d, loss=%.5f' % (
                                            args.model, commit, time_string(), step, loss))
                    log('%s, %s, %s, step=%d, loss=%.5f' % (args.model, commit, time_string(), step, loss))
                    log('Input: %s' % sequence_to_text(input_seq))

        except Exception as e:
            log('Exiting due to exception: %s' % e, slack=True)
            traceback.print_exc()
            coord.request_stop(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.path.expanduser('.'))
    parser.add_argument('--input', default='../data/train.txt')
    parser.add_argument('--vctk', default='')
    parser.add_argument('--ljspeech', default='', help="Related to preprocessed wav files")
    parser.add_argument('--librispeech', default='', help="Related to raw flac files (big corpus)")

    parser.add_argument('--model', default='tacotron')
    parser.add_argument('--name', help='Name of the run. Used for logging. Defaults to model name.')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--restore_step', type=int, help='Global step to restore from checkpoint.')
    parser.add_argument('--summary_interval', type=int, default=1000,
                        help='Steps between running summary ops.')
    parser.add_argument('--checkpoint_interval', type=int, default=1000,
                        help='Steps between writing checkpoints.')
    parser.add_argument('--slack_url', help='Slack webhook URL to get periodic reports.')
    parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
    parser.add_argument('--git', action='store_true', help='If set, verify that the client is clean.')
    parser.add_argument('--gpu', default=0, type=int, help='Select gpu for computation')
    parser.add_argument('--threads', default=1, type=int, help='Select number of threads for enqueue operation')
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)  # added available gpu
    run_name = args.name or args.model
    log_dir = os.path.join(args.base_dir, '..', 'logs', run_name)
    os.makedirs(log_dir, exist_ok=True)
    infolog.init(os.path.join(log_dir, 'train.log'), run_name, args.slack_url)
    hparams.parse(args.hparams)
    train(log_dir, args)


if __name__ == '__main__':
    main()
