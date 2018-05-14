# Neural Speech Project

The nspeech git repository is a collection of neural models for speech synthesis and neural vocoder.
It serves implementations of the following models:
- Tacotron
- Tacotron 2
- Wavenet


## Quick Start

### Installing dependencies

1. Install Python 3.

2. Install requirements:
   ```
   pip install -r requirements.txt
   ```

### Training

*Note: you need at least 50GB memory for the data to train a model.*

1. **Download and prepare datasets (if necessary) for training**

4. **Train TTS model (tacotron 1 and 2)**
   ```
   python3 train.py --ljspeech /data/LJSpeech-1.0/ --model taco1
   ```
   1. Select a graphics card by using `--gpu x` and the number of threads processing the data with `--threads y`
   
   2. To store summaries and checkpoints, set `--summary_interval x` and `--checkpoint_interval y`
   
   3. Tunable hyperparameters are found in the yaml files within the configs folder. You can adjust these at the command
   line using the `--hparams` flag, for example `--hparams="batch_size=16,outputs_per_step=2"`.
   Hyperparameters should generally be set to the same values at both training and eval time.

4. **Train neural vocoder model (wavenet)**
   Data processing differs compared with TTS training.
   ```
   python3 train_wavenet.py --ljspeech /data/LJSpeech-1.0/ --model simple_wavenet
   ```
   



5. **Monitor with Tensorboard** (optional)
   ```
   tensorboard --logdir ~/tacotron/logs-tacotron
   ```

   The trainer dumps audio and plots (alignment, wave, spectrogram) every 1000 steps (default). The are stored in the `log-dir`.

6. **Synthesize from a checkpoint**
   ```
   python3 demo_server.py --checkpoint ~/tacotron/logs-tacotron/model.ckpt-185000
   ```
   Replace "185000" with the checkpoint number that you want to use, then open a browser
   to `localhost:9000` and type what you want to speak. Alternately, you can
   run [eval.py](eval.py) at the command line:
   ```
   python3 eval.py --checkpoint ~/tacotron/logs-tacotron/model.ckpt-185000
   ```
   If you set the `--hparams` flag when training, set the same value here.


## Notes and Common Issues

  * You can train with [CMUDict](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) by downloading the
    dictionary to ~/tacotron/training and then passing the flag `--hparams="use_cmudict=True"` to
    train.py. This will allow you to pass ARPAbet phonemes enclosed in curly braces at eval
    time to force a particular pronunciation, e.g. `Turn left on {HH AW1 S S T AH0 N} Street.`

  * Occasionally, you may see a spike in loss and the model will forget how to attend (the
    alignments will no longer make sense). Although it will recover eventually, it may
    save time to restart at a checkpoint prior to the spike by passing the
    `--restore_step=150000` flag to train.py (replacing 150000 with a step number prior to the
    spike). **Update**: a recent [fix](https://github.com/keithito/tacotron/pull/7) to gradient
    clipping by @candlewill may have fixed this.
    
  * During eval and training, audio length is limited to `max_iters * outputs_per_step * frame_shift_ms`
    milliseconds. With the defaults (max_iters=200, outputs_per_step=5, frame_shift_ms=12.5), this is
    12.5 seconds.
    
    If your training examples are longer, you will see an error like this:
    `Incompatible shapes: [32,1340,80] vs. [32,1000,80]`
    
    To fix this, you can set a larger value of `max_iters` by passing `--hparams="max_iters=300"` to
    train.py (replace "300" with a value based on how long your audio is and the formula above).


## Other Implementations that influence this repository
  * By Keitho: https://github.com/keithito/tacotron
  * By Alex Barron: https://github.com/barronalex/Tacotron
  * By Kyubyong Park: https://github.com/Kyubyong/tacotron
