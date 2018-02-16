import librosa.display
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_alignment(alignment, path, info=None):
    fig, ax = plt.subplots()
    im = ax.imshow(
            alignment,
            aspect='auto',
            origin='lower',
            interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.savefig(path, format='png')


def plot_specgram(spec, path, spec_type="linear"):
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(spec, y_axis=spec_type)
    plt.colorbar(format='%+2.0f dB')
    if spec_type == "linear":
        plt.title('Linear-frequency power spectrogram')
    elif spec_type == "mel":
        plt.title('Mel spectrogram')
    else:
        # TODO: default title?
        pass
    plt.tight_layout()
    plt.savefig(path, format='png')
