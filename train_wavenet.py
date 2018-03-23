import matplotlib
import tensorflow as tf

from datasets.WavenetDataFeeder import WavenetDataFeeder

matplotlib.use('Agg')

import argparse
import math
import os
import time
import traceback

import models
from util import ValueWindow, infolog
from util.infolog import log

HPARAMS = tf.contrib.training.HParams(
        # Comma-separated list of cleaners to run on text prior to training and eval. For non-English
        # text, you may want to use "basic_cleaners" or "transliteration_cleaners" See TRAINING_DATA.md.
        cleaners='english_cleaners',

        # Audio:
        num_mels=80,
        num_freq=1025,  # 2048,
        sample_rate=16000,  # 24000,
        frame_length_ms=50,
        frame_shift_ms=12.5,
        preemphasis=0.97,
        min_level_db=-100,
        ref_level_db=20,

        # custom
        silence_threshold=0.1,

        # Model:
        outputs_per_step=5,
        filter_width=2,
        dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                   1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                   1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                   1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                   1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        # dilations=[1, 2, 4, 8, 16, 32],
        residual_channels=32,
        # residual_channels=8,
        dilation_channels=32,
        # dilation_channels=8,
        quantization_channels=256,
        # quantization_channels=64,
        skip_channels=512,
        # skip_channels=128,
        use_biases=False,
        scalar_input=False,
        initial_filter_width=32,
        gc_channels=16,  # speaker embedding size
        gc_category_cardinality=276,  # maximum speaker id
        lc_channels=0,
        l2_regularization_strength=0,

        # Training:
        batch_size=1,
        batch_group_size=1,
        queue_size=32,  # number of batches stored in queue
        min_dequeue_ratio=0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        initial_learning_rate=0.002,
        learning_rate_decay_halflife=100000,
        decay_learning_rate=True
)

tf_config = tf.ConfigProto(log_device_placement=False)

from train import prepare_input_paths


def hparams_debug_string():
    values = HPARAMS.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)


def train_wavenet(log_dir, args):
    checkpoint_path = os.path.join(log_dir, 'model.ckpt')

    input_paths = prepare_input_paths(args)

    log('Checkpoint path: %s' % checkpoint_path)
    log('Loading training data from: %s' % input_paths)
    log('Using model: %s' % args.model)
    log(hparams_debug_string())
    hp = HPARAMS

    with tf.Session(config=tf_config) as sess:

        # Create coordinator.
        coord = tf.train.Coordinator()

        # Load raw waveform from VCTK corpus.
        # with tf.name_scope('create_inputs'):
        # Allow silence trimming to be skipped by specifying a threshold near
        # zero.
        # silence_threshold = args.silence_threshold if args.silence_threshold > \
        #                                               EPSILON else None
        # gc_enabled = args.gc_channels is not None
        # reader = AudioReader(
        #     args.data_dir,
        #     coord,
        #     sample_rate=wavenet_params['sample_rate'],
        #     gc_enabled=gc_enabled,
        #     receptive_field=WaveNetModel.calculate_receptive_field(wavenet_params["filter_width"],
        #                                                            wavenet_params["dilations"],
        #                                                            wavenet_params["scalar_input"],
        #                                                            wavenet_params["initial_filter_width"]),
        #     sample_size=args.sample_size,
        #     silence_threshold=silence_threshold)
        # audio_batch = reader.dequeue(args.batch_size)
        # if gc_enabled:
        #     gc_id_batch = reader.dequeue_gc(args.batch_size)
        # else:
        #     gc_id_batch = None

        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Create network.
        model = models.create_model("wavenet", hp)

        with tf.variable_scope('datafeeder'):
            feeder = WavenetDataFeeder(sess, coord, input_paths, model.receptive_field, hp)

        if hp.l2_regularization_strength == 0:
            hp.l2_regularization_strength = None
        # TODO
        model.initialize(feeder.audio, feeder.speaker_ids)#, feeder.mel_targets)
        model.add_loss()
        # ,
        # global_condition_batch=gc_id_batch,
        # l2_regularization_strength=args.l2_regularization_strength)
        model.add_optimizer(global_step)
        model.add_stats()

        # Set up logging for TensorBoard.
        # writer = tf.summary.FileWriter(logdir)
        # writer.add_graph(tf.get_default_graph())
        # run_metadata = tf.RunMetadata()
        # summaries = tf.summary.merge_all()

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
                    # log('Saving audio and alignment...')
                    # compute one example
                    # input_seq, spectrogram, alignment = sess.run([
                    #         model.inputs[0], model.linear_outputs[0], model.alignments[0]])
                    #     waveform = audio.inv_spectrogram(spectrogram.T)
                    #     audio.save_wav(waveform, os.path.join(log_dir, 'step-%d-audio.wav' % step))
                    #     # np.save(os.path.join(log_dir, 'step-%d-align.npy' % step), alignment)
                    #     plot.plot_alignment(alignment, os.path.join(log_dir, 'step-%d-align.png' % step),
                    #                         info='%s, %s, %s, step=%d, loss=%.5f' % (
                    #                             args.model, commit, time_string(), step, loss))
                    #     log('%s, %s, %s, step=%d, loss=%.5f' % (args.model, commit, time_string(), step, loss))
                    #     log('Input: %s' % sequence_to_text(input_seq))

        except Exception as e:
            log('Exiting due to exception: %s' % e, slack=True)
            traceback.print_exc()
            coord.request_stop(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.path.expanduser('.'))
    parser.add_argument('--vctk', default='')
    parser.add_argument('--ljspeech', default='', help="Related to preprocessed wav files")
    parser.add_argument('--librispeech', default='', help="Related to raw flac files (big corpus)")

    parser.add_argument('--model', default='wavenet')
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
    HPARAMS.parse(args.hparams)
    train_wavenet(log_dir, args)


if __name__ == '__main__':
    main()
