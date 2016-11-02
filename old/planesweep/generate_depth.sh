#!/usr/bin/env bash

OUT_DIR="/Volumes/Bahia/videoplanes"

mkdir -p $OUT_DIR

for DEPTH_INT in {0..255}
do
    DEPTH_BYTE=$(printf "%.2x" $DEPTH_INT)

    DEPTH_COLOR="#$DEPTH_BYTE$DEPTH_BYTE$DEPTH_BYTE"

    echo "Generating video with constant 8 bit depth: $DEPTH_INT/255"
    echo "Matching frame color: $DEPTH_COLOR"

    convert -size 1024x768 xc:$DEPTH_COLOR $OUT_DIR/depth-$DEPTH_INT.png
    ffmpeg -loop 1 -i $OUT_DIR/depth-$DEPTH_INT.png -c:v rawvideo -r 1 -t 100 -pix_fmt y8 $OUT_DIR/depth-$DEPTH_INT.yuv
    rm $OUT_DIR/depth-$DEPTH_INT.png
done
