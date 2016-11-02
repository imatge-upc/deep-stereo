#!/bin/bash
echo "Running experiments... depth estimation!"

module load cuda/7.5 cudnn/v4 opencv/3.1.0 leptonica tesseract protobuf openni2 vtk
source ~/tensorflow_env/bin/activate
echo "Activated virtualenv"

export PYTHONPATH=~/work/depthestimation:$PYTHONPATH
python tf_deep_stereo/tf_train.py --batch-size 40 --patches-per-set 5 --extraction-workers 10 --aggregation-workers 2
