#!/bin/bash

srun -c 10 --reservation=TensorFlow --pty --gres=gpu:1 --mem=80000 run_experiment.sh
