from neural_speech.models.tacotron import Tacotron
from neural_speech.models.tacotron2 import Tacotron2
from neural_speech.models.wavenet import WaveNetModel
from neural_speech.models.wavenet_simple import SimpleWaveNetModel


def create_model(name, hparams):
    if name == 'taco1':
        return Tacotron(hparams)
    if name == 'taco2':
        return Tacotron2(hparams)
    if name == 'wavenet':
        return WaveNetModel(hparams)
    if name == 'simple_wavenet':
        return SimpleWaveNetModel(hparams)
    else:
        raise Exception('Unknown model: ' + name)
