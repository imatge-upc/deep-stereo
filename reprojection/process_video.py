
from PIL import Image
from camera_parser import CameraParser
from yuv_reader import YUVReader
from reprojection import Reprojection
from optparse import OptionParser
from six.moves import cPickle
import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    parser = OptionParser()
    parser.add_option("-o", "--camera-original", dest="cameraOriginalFile",
                      help="Original camera file to load intrinsic matrix from", metavar="FILE")
    parser.add_option("-v", "--camera-virtual", dest="cameraVirtualFile",
                      help="Virtual camera file to load intrinsic matrix from", metavar="FILE")
    parser.add_option("-i", "--input", dest="video",
                      help="video file to process", metavar="FILE")

    parser.add_option("-r", "--result", dest="output",
                      help="Save output images on that path")
    parser.add_option("-p", "--prefix", dest="prefix", default="depth",
                      help="Output prefix")

    parser.add_option("-g", "--generate-images", dest="generate_images", default=False, action="store_true",
                      help="Generate PNG images for each depth plane")

    (options, args) = parser.parse_args()

    # Create output dir if it does not exist
    if not os.path.exists(options.output):
        os.makedirs(options.output)

    # Read camera parameters from cameraFile argument
    print "Original camera file: %s" % options.cameraOriginalFile
    print "Virtual camera file: %s" % options.cameraVirtualFile

    cam_original = CameraParser(options.cameraOriginalFile)
    cam_virtual = CameraParser(options.cameraVirtualFile)

    # Get a frame for YUV video
    print "Load YUV (i480) video file: %s" % options.video
    video = YUVReader(options.video, (1024, 768))
    frame = video.getRGBFrame(0)


    # Reproject at all needed depths
    print "Saving plane sweep volume: %s" % options.output
    r = Reprojection(cam_original, frame)

    # Reproject at depths 1m to 200m in steps of 2 meters (total 100 images)
    volume_array = []
    i = 0
    depth_from = 1
    depth_to = 200
    depth_step = 2
    for d in xrange(depth_from, depth_to, depth_step):
        #print "Calculate projection at depth: %s" % d
        result_prj = r.reprojectTo(cam_virtual, depth=d)
        volume_array.append(result_prj)

        # if the generate image is selected then output an image for each depth plane
        if options.generate_images:
            im = Image.fromarray(result_prj, 'RGBA')
            path = os.path.join(options.output, "%s_%03d_%03d.png" % (options.prefix, d, i))
            print "Saving image: %s" % path
            im.save(path)
        i += 1
    print "Depth sweep plane calculation done."

    # Save plane sweep volume in a cPickle file with stacked images
    if not options.generate_images:
        print "Saving pickle file..."
        psw_volume = np.dstack(volume_array)
        vol_name = os.path.join(options.output, "psw_%s_%s.pkl" % (depth_from, depth_to))
        cPickle.dump(psw_volume, open(vol_name, "wb"))
        print "Plane sweep volume file saved in %s" % vol_name


if __name__ == "__main__": main()