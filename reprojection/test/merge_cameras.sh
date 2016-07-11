#!/bin/bash

DIRCAM_A=$1
DIRCAM_B=$2
OUTDIR=$3

mkdir -p $OUTDIR

echo "Compsing files from:"
echo "Camera A: $DIRCAM_A"
echo "Camera B: $DIRCAM_B"

CAM_A=($( ls $DIRCAM_A ))
CAM_B=($( ls $DIRCAM_B ))

for i in "${!CAM_A[@]}" 
do
	IMAGE_1=$DIRCAM_A/${CAM_A[$i]}
	IMAGE_2=$DIRCAM_B/${CAM_B[$i]}
	COMPOSED_OUT="$OUTDIR/composed_$i.jpg"
	echo "Generating image ($i) from:"
	echo -e "\t Cam A: $IMAGE_1"
	echo -e "\t Cam B: $IMAGE_2"
	echo -e "\t Output: $COMPOSED_OUT"
	composite -blend 50x50 -gravity center $IMAGE_1 $IMAGE_2 -alpha Set $COMPOSED_OUT
#	exit
done

