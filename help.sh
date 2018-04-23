#!/usr/bin/env bash

# synchronize local data and remote
rsync -acvz --progress neural_speech/* nspeech1:project/neural_speech_dev/

# run training with ljspeech
python3 train.py --ljspeech /data/LJSpeech-1.0/ --model taco1 --name logs/taco1-lj --summary_interval 100 --checkpoint_interval 100 --threads 8 --gpu 0

# rerun training with ljspeech
python3 train.py --ljspeech /data/LJSpeech-1.0/ --model taco1 --name logs/taco1-lj --summary_interval 100 --checkpoint_interval 100 --threads 8 --gpu 0 --restore_step 2100

python3 train_wavenet.py --ljspeech /data/LJSpeech-1.0/ --model wavenet --name wavenet-test --summary-interval 1000 --checkpoint-interval 1000 --threads 2 --gpu 0