from models.tacotron import Tacotron
from models.tacotron2 import Tacotron2


def create_model(name, hparams):
    if name == 'taco1':
        return Tacotron(hparams)
    if name == 'taco2':
        return Tacotron2(hparams)
    else:
        raise Exception('Unknown model: ' + name)
