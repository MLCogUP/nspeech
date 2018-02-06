from models.tacotron import Tacotron
from models.tacotron2 import Tacotron2


def create_model(name, hparams):
    if name == 'tacotron':
        return Tacotron(hparams)
    if name == 'tacotron2':
        return Tacotron2(hparams)
    else:
        raise Exception('Unknown model: ' + name)
