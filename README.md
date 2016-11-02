## Useful commands

### Play ballet depht camera
    mplayer -demuxer rawvideo -rawvideo w=1024:h=768:format=y8 ballet-depth_2.yuv

### Play ballet color camera
    mplayer -demuxer rawvideo -rawvideo w=1024:h=768:format=i420 ballet-color_2.yuv

### Generate solid green image 
    convert -size 1024x768 canvas:green canvas_green.png
### Generate solid black image
    convert -size 1024x768 xc:#FFFFFF red.png

### Generate green canvas video to display
    ffmpeg -loop 1 -i canvas_green.png -c:v libx264 -t 30 -pix_fmt yuv420p out.mp4
    
### Generate a 100 frame video using image on canvas_green.png
    ffmpeg -loop 1 -i canvas_green.png -c:v rawvideo -r 1 -t 100 -pix_fmt yuv420p out.yuv
    
## Note: Change -pix_fmt to `y8` to convert depth estimation videos
    
    
### Use 1 gpus in caffe
    --gres=gpu:1
    
    
## WHICH CNN Frameworks to use:
    * Theanos
    * Keras (Prefered, new)
    * Tensorflow (from Google)
    
## Extract 1 frame from undo dancer seq
    ffmpeg -ss 00:00:05 -t 1 -s 1920x1088 -i Dancer_c_2_1920x1088.yuv -f mjpeg Dancer_c2_frame.jpg
    
## Camera merge test sequences for plane sweep volume
    ./merge_cameras.sh ../../sweep_results_test/cam0 ../../sweep_results_test/cam1 ../../sweep_results_test/merged

## Search for variables defined in TF variable scope
    [n.op.name for n in tf.all_variables()]