import tensorflow as tf
import yaml

yaml_path = "neural_speech/hparams/"
_hparams = None


def debug_string(hp):
    values = hp.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)


def load(model_type):
    global _hparams
    audio_config = yaml.load(open(yaml_path + "audio.yaml"))
    train_config = yaml.load(open(yaml_path + "train.yaml"))
    model_config = yaml.load(open(yaml_path + model_type + ".yaml"))
    audio_config.update(train_config)
    audio_config.update(model_config)
    _hparams = tf.contrib.training.HParams(**audio_config)
    return _hparams


def get_hparams():
    return _hparams
