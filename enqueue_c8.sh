#!/bin/bash

srun -c 10 --pty --gres=gpu:2 --mem=60000 run_experiment_c8.sh
