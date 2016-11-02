from reprojection.camera import Camera
import numpy as np


class BalletCamera(Camera):

    def __init__(self, filename):
        """
        [Image size:]
        1024
        768

        [Rotation matrix:]
        0.949462	0.046934	0.310324
        -0.042337	0.998867	-0.021532
        -0.310985	0.007308	0.950373

        [Translation vector:]
        -15.094651	0.189829	1.383263

        [Calibration matrix:]
        1918.270000	2.489820	494.085000
        0.000000	1922.580000	447.736000
        0.000000	0.000000	1.000000

        [Projection matrix:]
        1667.566036	96.129856	1064.796652	-28271.694034
        -220.635449	1923.673772	384.119213	984.298081
        -0.310985	0.007308	0.950373	1.383263

        [Distortion parameters:]
        1.383263
        0.000000
        0.000000
        0.000000
        """
        with open(filename, 'r') as f:
            lines = f.readlines()

            # Parse rotation
            r1 = [float(i) for i in lines[5].split('\t')[0:3]]
            r2 = [float(i) for i in lines[6].split('\t')[0:3]]
            r3 = [float(i) for i in lines[7].split('\t')[0:3]]
            rotation = np.array([r1, r2, r3])

            # Parse intrinsics
            k1 = [float(i) for i in lines[13].split('\t')[0:3]]
            k2 = [float(i) for i in lines[14].split('\t')[0:3]]
            k3 = [float(i) for i in lines[15].split('\t')[0:3]]
            intrinsics = np.array([k1, k2, k3])

            # Parse translation
            translation = [float(i) for i in lines[10].split('\t')[0:3]]

        super(BalletCamera, self).__init__(rotation, translation, intrinsics)

class DepthEncoder:
    """
    To convert the intensity image into z or depth values use the following equation:
    z(r,c) = 1.0/((P(r,c)/255.0)*(1.0/MinZ - 1.0/MaxZ) + 1.0/MaxZ);

    where z(r,c) is the z or depth value (in x,y,z coordinates, the optical center of
    the 5th camera is the origin of the world coordinates.)
    P(r,c) is the intensity value and MinZ is 42.0 and MaxZ is 130.0.
    """
    @staticmethod
    def intensity_to_meters(intensity):

        MinZ = 42.0
        MaxZ = 130.0
        meters = 1.0 / ((intensity/255.0) * (1.0/MinZ - 1.0/MaxZ) + 1.0/MaxZ)
        return meters

    @staticmethod
    def meters_to_intensity(meters):

        MinZ = 42.0
        MaxZ = 130.0
        intensity = (1.0/meters - 1.0/MaxZ) / (1.0/MinZ - 1.0/MaxZ) * 255.0
        return intensity


    @staticmethod
    def test_metrics():
        intensity = 10.0
        meters = DepthEncoder.intensity_to_meters(intensity)

        print "Intensity to meters: (%s) -> (%s meters)" % (intensity, meters)

        intensity = DepthEncoder.meters_to_intensity(meters)

        print "Meters to intensity: (%s meters) -> (%s)" % (meters, intensity)


