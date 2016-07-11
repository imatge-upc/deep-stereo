import numpy as np
from numpy.linalg import inv
import cv2
from PIL import Image



class Reprojection(object):

    def __init__(self, cameraOriginal, frameOriginal, verbose=False):
        self.verbose = verbose
        self.camera = cameraOriginal
        self.frame = frameOriginal

        self.w = self.frame.shape[1]
        self.h = self.frame.shape[0]

        # Original points to reproject
        self.src_pts = np.reshape([0, 0,
                                   self.w, 0,
                                   self.w, self.h,
                              0, self.h], (4, 2))

    def reprojectTo(self, cameraVirtual, depth):

        P_1 = self.camera.getProjection()

        P_i_33_1 = inv(P_1[:3, :3])
        P_4_1 = P_1[:, 3]

        # Backproject points to certain depth on camera plane
        pts_3D = np.zeros([4, 4])
        i = 0
        for x in self.src_pts:
            A = np.append(-P_i_33_1.dot(P_4_1), 1)
            x_h = np.append(x, 1)
            B = np.append(P_i_33_1.dot(x_h), 0)
            X = A + depth * B
            if self.verbose:
                print "(Backprojection) Point in image-> %s Point in 3D-> %s" % (x, X)
            pts_3D[i, :] = X
            i += 1

        if self.verbose:
            print pts_3D

        # Reproject 3D points back to second camera plane
        dst_pts = np.zeros([4, 2])
        i = 0
        for X in pts_3D:
            x_h = cameraVirtual.getProjection().dot(X)
            # Back from homogeneous coordinates
            x = np.divide(x_h, x_h[2])[:2]
            if self.verbose:
                print "(Projection ) Point 3D-> %s point in image -> %s" % (X, x)
            dst_pts[i, :] = x
            i += 1

        if self.verbose:
            print dst_pts

        # Calculate 2D homograpy and convert image
        M, mask = cv2.findHomography(self.src_pts, dst_pts, cv2.RANSAC, 5.0)
        #self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2RGBA)
        b_channel, g_channel, r_channel = cv2.split(np.asarray(self.frame))
        alpha_channel = np.ones((self.h, self.w), dtype=np.uint8) * 255
        img_RGBA = np.dstack((b_channel, g_channel, r_channel, alpha_channel))
        result = cv2.warpPerspective(img_RGBA, M, (self.w, self.h))
        return result

