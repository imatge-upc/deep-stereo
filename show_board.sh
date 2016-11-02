#!/bin/bash
source ~/tensorflow_env/bin/activate
module load cuda
tensorboard --port=5555 --logdir /imatge/mpomar/work/tf_train
