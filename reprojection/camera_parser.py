
import numpy as np

class Camera(object):

    def __init__(self, rotation, translation, intrinsics):

        if rotation.shape[0] != 3 and rotation.shape[1] != 3:
            raise NameError("Rotation matrix must shape must be (3,3)")

        if intrinsics.shape[0] != 3 and intrinsics.shape[1] != 3:
            raise NameError("Intrinsics matrix must shape must be (3,3)")

        self.rotation_matrix = rotation
        self.translation = translation
        self.intrinsics = intrinsics

    def getRotation(self):
        return self.rotation_matrix

    def getTranslation(self):
        return self.translation

    def getIntrinsics(self):
        return self.intrinsics

    def getProjection(self):
        RT = np.concatenate((self.getRotation(), np.vstack(self.getTranslation())), axis=1)
        return self.getIntrinsics().dot(RT)


class CameraParser(Camera):

    def __init__(self, fileName):
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
        with open(fileName, 'r') as f:
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

        super(CameraParser, self).__init__(rotation, translation, intrinsics)


