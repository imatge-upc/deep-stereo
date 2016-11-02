import numpy as np
import cv2
from dataset_preparation.kitti_camera import Camera

class Reprojection(object):

    def __init__(self, camSrc, camVirtual, verbose=False):
        if not isinstance(camSrc, Camera) or not isinstance(camVirtual, Camera):
            raise ValueError("camSrc and camVirtual must be Camera objects")

        self.verbose = verbose
        self.camSRC = camSrc
        self.camVIRTUAL = camVirtual

    def reproject(self, depth, frame):
        #print "Reprojecting at depth: %s" % depth
        w = frame.shape[1]
        h = frame.shape[0]
        # 4 Corners on the virtual camera to get te 4 rays that intersect with the depth plane
        src_pts = np.reshape([
            0, 0,
            w, 0,
            w, h,
            0, h], (4, 2))

        dst_pts = np.ones((4, 2))
        i = 0
        for p in src_pts:
            #print p

            #print self.camSRC.intrinsics
            c = self.camVIRTUAL.get_principal_point()
            f = self.camVIRTUAL.get_focal_length()

            # Point in virtual camera to corresponding depth
            #            Pc(1) = (x - cx) * z(r, c) / fx;
            #            Pc(2) = (y - cy) * z(r, c) / fy;
            #            Pc(3) = z(r, c);
            pH = np.asarray([
                (p[0] - c[0]) * depth / f[0],
                (p[1] - c[1]) * depth / f[1],
                depth,
            ])

            # Convert point to world coordinates (CUIDADO CON ESTO QUE ES EL
            # PROBLEMA DE QUE R Y T ESTEN DEFINIDAS DE MUNDO A CAMARA O AL
            # REVES
            #Pw = R'*(Pc-T); % EC. SI R,T estan definidas de mundo a camara (no es el caso para undo dancer)
            #Pw = R*PH+T; # EC. si R,T definidas de camara a mundo (caso undo dancer)
            #Pw(4) = 1;

            pW = self.camVIRTUAL.getRotation().dot(pH) + self.camVIRTUAL.getTranslation()
            pW = np.append(pW, 1.0) # to homogeneous coordinates
            #print pW

            # Reproject back to second camera
            #pix = cams{2}.K * cams{2}.E * Pw;
            pix = reduce(np.dot, [self.camSRC.getIntrinsics(), self.camSRC.get_extrinsics_inv(), pW])
            #print pix
            dst = np.asarray([pix[0]/pix[2], pix[1]/pix[2]])
            #print dst
            # homog to normal coords
            dst_pts[i, :] = dst
            i += 1

        #print "Source points"
        #print src_pts
        #print "Reprojected at depth (%s)" % depth
        #print dst_pts

        return Reprojection.do_homography(dst_pts, src_pts, frame, (w, h))

    @staticmethod
    def do_homography(src_pts, dst_pts, frame, size):
        """
        ENSURE OPENCV version is at least 3.1.0
        :param src_pts:
        :param dst_pts:
        :param frame:
        :param size:
        :return:
        """
        #print("Find homography from: %s to %s" % (src_pts, dst_pts))
        w = size[0]
        h = size[1]

        # Ensure we passed proper points
        assert src_pts.shape[0] == 4 and src_pts.shape[1] == 2
        assert dst_pts.shape[0] == 4 and dst_pts.shape[1] == 2

        # Calculate 2D homograpy and convert image
        M, mask = cv2.findHomography(src_pts, dst_pts)

        # Add alpha channel to act as a mask for not existing pixels
        alpha_channel = np.ones((h, w), dtype=np.float32)
        img_RGBA = np.dstack((frame, alpha_channel))

        #print("reproject image")
        result = cv2.warpPerspective(img_RGBA, M, (w, h))
        #print("Reprojected image done")
        return result

