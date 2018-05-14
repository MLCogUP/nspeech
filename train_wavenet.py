import argparse
import math
import os
import time
import traceback

import tensorflow as tf

import neural_speech.hparams
import neural_speech.models
from neural_speech.datasets.WavenetDataFeeder import WavenetDataFeeder
from neural_speech.utils import ValueWindow, infolog
from neural_speech.utils.infolog import log
from train import prepare_input_paths

tf_config = tf.ConfigProto(log_device_placement=False)


def train_wavenet(log_dir, args, hp):
    checkpoint_path = os.path.join(log_dir, 'model.ckpt')

    input_paths = prepare_input_paths(args)

    log('Checkpoint path: %s' % checkpoint_path)
    log('Loading training data from: %s' % input_paths)
    log('Using model: %s' % args.model)
    log(neural_speech.hparams.debug_string(hp))

    with tf.Session(config=tf_config) as sess:

        # Create coordinator.
        coord = tf.train.Coordinator()

        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Create network.
        model = neural_speech.models.create_model(args.model, hp)

        with tf.variable_scope('datafeeder'):
            feeder = WavenetDataFeeder(sess, coord, input_paths, model.receptive_field, hp)
        hp.num_speakers = len(feeder.speaker2id)
        hp.gc_category_cardinality = hp.num_speakers

        if hp.l2_regularization_strength == 0:
            hp.l2_regularization_strength = None

        global_condition = feeder.speaker_ids if hp.gc_channels > 0 else None
        local_condition = feeder.mel_targets if hp.lc_channels > 0 else None

        model.initialize(feeder.audio, global_condition, local_condition)
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
                log('Starting new training run ', slack=True)

            feeder.start_threads(args.threads)

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

        except Exception as e:
            log('Exiting due to exception: %s' % e, slack=True)
            traceback.print_exc()
            coord.request_stop(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', default=os.path.expanduser('logs'))
    parser.add_argument('--vctk', default='')
    parser.add_argument('--ljspeech', default='', help="Related to preprocessed wav files")
    parser.add_argument('--librispeech', default='', help="Related to raw flac files (big corpus)")

    parser.add_argument('--model', default='wavenet')
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
    HPARAMS = neural_speech.hparams.load("wavenet")
    HPARAMS.parse(args.hparams)
    train_wavenet(log_dir, args, HPARAMS)


if __name__ == '__main__':
    main()
