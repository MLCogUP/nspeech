import argparse
import math
import os
import time
import traceback

import joblib
import tensorflow as tf

import neural_speech.hparams
from neural_speech.datasets.datafeeder import DataFeeder
from neural_speech.models import create_model
from neural_speech.utils import audio, ValueWindow, plot, time_string, infolog
from neural_speech.utils.infolog import log
from neural_speech.utils.text import sequence_to_text


def prepare_input_paths(args):
    input_paths = {}
    if args.vctk:
        input_paths["vctk"] = args.vctk
    if args.ljspeech:
        input_paths["ljspeech"] = args.ljspeech
    if args.librispeech:
        input_paths["librispeech"] = args.librispeech

    return input_paths


def train(log_dir, args, hparams):
    checkpoint_path = os.path.join(log_dir, 'model.ckpt')

    input_paths = prepare_input_paths(args)

    log('Checkpoint path: %s' % checkpoint_path)
    log('Loading training data from: %s' % input_paths)
    log('Using model: %s' % args.model)
    log(neural_speech.hparams.debug_string(hparams))

    with tf.Session() as sess:
        # Set up DataFeeder:
        coord = tf.train.Coordinator()
        with tf.variable_scope('datafeeder'):
            feeder = DataFeeder(sess, coord, input_paths, hparams)
        hparams.num_speakers = len(feeder.speaker2id)

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
                log('Resuming from checkpoint: %s' % restore_path, slack=True)
            else:
                log('Starting new training run', slack=True)

            feeder.start_threads(args.threads)
            log('Feeder started')

            while not coord.should_stop():
                start_time = time.time()
                step, loss, opt, qsize = sess.run([global_step, model.loss, model.optimize, feeder.size])
                time_window.append(time.time() - start_time)
                loss_window.append(loss)
                message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f, queue=%.02f]' % (
                    step, time_window.average, loss, loss_window.average, (qsize / float(feeder.capacity)))
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
                    input_seq, spectrogram, melgram, alignment, wav = sess.run([
                        model.inputs[0], model.linear_outputs[0], model.mel_outputs[0], model.alignments[0],
                        model.audio[0]])
                    # TODO: replace with gpu griffinlim impl
                    # waveform = audio.inv_spectrogram(spectrogram.T)

                    waveform = audio.inv_preemphasis(wav)
                    waveform = waveform[:audio.find_endpoint(waveform)]
                    audio.save_wav(waveform, os.path.join(log_dir, 'step-{:06}-audio.wav'.format(step)))
                    # np.save(os.path.join(log_dir, 'step-%d-align.npy' % step), alignment)
                    plot.plot_alignment(alignment, os.path.join(log_dir, 'step-{:06}-align.png'.format(step)),
                                        info='%s, %s, step=%d, loss=%.5f' % (
                                            args.model, time_string(), step, loss))
                    plot.plot_wave(waveform, hparams.sample_rate,
                                   os.path.join(log_dir, 'step-{:06}-wav.png'.format(step)),
                                   sequence_to_text(input_seq))
                    plot.plot_specgram(spectrogram, os.path.join(log_dir, 'step-{:06}-lin.png'.format(step)), "linear")
                    plot.plot_specgram(melgram, os.path.join(log_dir, 'step-{:06}-mel.png'.format(step)), "mel")
                    log('%s, %s, step=%d, loss=%.5f' % (args.model, time_string(), step, loss))
                    log('Input: %s' % sequence_to_text(input_seq))

                    # TODO remove feeder dump - replace by separate preprocessor again - also h5py!
                    # dumps feeder iff no changes are recognized
                    if feeder.dump_status == 2:
                        joblib.dump(feeder.processed_data, "/cache/data.joblib", compress=0)
                        feeder.dump_status = 3

        except Exception as e:
            log('Exiting due to exception: %s' % e, slack=True)
            traceback.print_exc()
            coord.request_stop(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', default=os.path.expanduser('logs'))
    parser.add_argument('--input', default='../data/train.txt')
    parser.add_argument('--vctk', default='')
    parser.add_argument('--ljspeech', default='', help="Related to preprocessed wav files")
    parser.add_argument('--librispeech', default='', help="Related to raw flac files (big corpus)")

    parser.add_argument('--model', default='taco1')
    parser.add_argument('--name', help='Name of the run. Used for logging. Defaults to model name.')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--restore-step', type=int, help='Global step to restore from checkpoint.')
    parser.add_argument('--summary-interval', type=int, default=1000,
                        help='Steps between running summary ops.')
    parser.add_argument('--checkpoint-interval', type=int, default=1000,
                        help='Steps between writing checkpoints.')
    parser.add_argument('--slack-url', help='Slack webhook URL to get periodic reports.')
    parser.add_argument('--tf-log-level', type=int, default=1, help='Tensorflow C++ log level.')
    parser.add_argument('--git', action='store_true', help='If set, verify that the client is clean.')
    parser.add_argument('--gpu', default=0, type=int, help='Select gpu for computation')
    parser.add_argument('--threads', default=1, type=int, help='Select number of threads for enqueue operation')
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)  # added available gpu
    run_name = args.name or args.model
    log_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    infolog.init(os.path.join(log_dir, 'train.log'), run_name, args.slack_url)
    hparams = neural_speech.hparams.load(args.model)
    hparams.parse(args.hparams)
    train(log_dir, args, hparams)


if __name__ == '__main__':
    main()
