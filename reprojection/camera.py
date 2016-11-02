import numpy as np
import math

class Camera(object):

    def __init__(self, rotation, translation, intrinsics, name="camera"):
        self.name = name

        if len(translation) != 3:
            raise NameError("Translation vector must shape must be of length 3")

        if rotation.shape[0] != 3 and rotation.shape[1] != 3:
            raise NameError("Rotation matrix must shape must be (3,3)")

        if intrinsics.shape[0] != 3 and intrinsics.shape[1] != 3:
            raise NameError("Intrinsics matrix must shape must be (3,3)")

        self.rotation = rotation
        self.translation = translation
        self.intrinsics = intrinsics

    def get_principal_point(self):
        return np.asarray([self.intrinsics[0, 2], self.intrinsics[1, 2]])

    def get_focal_length(self):
        return np.asarray([self.intrinsics[0, 0], self.intrinsics[1, 1]])

    def get_extrinsics(self):
        return np.concatenate((self.rotation, np.vstack(self.translation)), axis=1)

    def get_extrinsics_inv(self):
        T_p = -np.transpose(self.rotation).dot(self.translation)
        return np.concatenate((np.transpose(self.rotation), np.vstack(T_p)), axis=1)

    def getRotation(self):
        return self.rotation

    def getTranslation(self):
        return self.translation

    def getIntrinsics(self):
        return self.intrinsics

    def getProjection(self):
        RT = np.concatenate((self.getRotation(), np.vstack(self.getTranslation())), axis=1)
        return self.getIntrinsics().dot(RT)

    def __repr__(self):
        return "%s" % self.getProjection()

def rotateX(angle):
    """ Rotates the point around the X axis by the given angle in degrees. """
    phi = angle * math.pi / 180
    return np.array([
        [1, 0, 0],
        [0, math.cos(phi), -math.sin(phi)],
        [0, math.sin(phi), math.cos(phi)]
    ])

MockCameraA = Camera(rotateX(-25), [0, 3, 0], np.eye(3))
MockCameraB = Camera(rotateX(25), [0, -3, 0], np.eye(3))
