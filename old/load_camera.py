import numpy as np
from enum import Enum
from glumpy import gloo, glm
import math
from glumpy.transforms.rotate import _rotation

np.set_printoptions(suppress=True)

class CameraRigType(Enum):
    Left = "1-Left"
    Virtual = "2-Virtual"
    Right = "3-Right"

class Camera(object):

    def __init__(self, intrinsics, rotation, translation, nx=1024, ny=768):
        self.nx = nx
        self.ny = ny
        self.intrinsics = np.reshape(intrinsics, (3, 3))
        self.rotation = np.reshape(rotation, (3, 3))
        self.translation = translation


    def get_principal_point(self):
        return np.array([self.intrinsics[0, 2], self.intrinsics[1, 2]])

    def get_focal_length(self):
        return np.array([self.intrinsics[0, 0], self.intrinsics[1, 1]])

    def get_skew(self):
        return self.intrinsics[0, 1]

    def get_extrinsics(self):
        return np.concatenate((self.rotation, np.vstack(self.translation)), axis=1)

    @property
    def rotationX(self):
        return math.atan2(self.rotation[2,1], self.rotation[2,2])

    @property
    def rotationY(self):
        return math.atan2(-self.rotation[2,0], math.sqrt((self.rotation[2,1]**2)+(self.rotation[2,2]**2)))

    @property
    def rotationZ(self):
        return math.atan2(self.rotation[1,0], self.rotation[0,0])

    @property
    def rotationAxis(self):
        return np.array([self.rotationX, self.rotationY, self.rotationZ])


    def camera_plane_at(self, depth=50.0):
        #p = np.array([
        #    (-1, 1, d),
        #    (1, 1, d),
        #    (1, -1, d),
        #    (-1, -1, d),
       # ], dtype=float)

        center = self.get_principal_point()
        CX = center[0]/2.0
        CY = center[1]/2.0

        #points = np.array([
        #        -CX,                     -CY,    depth,
        #        self.nx - 1-CX,          -CY,    depth,
        #        self.nx - 1-CX, self.ny-1-CY,    depth,
        #        -CX,            self.ny-1-CY,    depth
        #])

        points = np.array([
                (-1, 1, depth),
                (1, 1, depth),
                (1, -1, depth),
                (-1, -1, depth),
        ])
        points = np.reshape(points, (4, 3))


        # Face Normals
        n = np.array([[0, 0, 1], [0, 0, 1]])

        # Texture coords
        t = np.array([
            [1, 0],
            [0, 0],
            [0, 1],
            [1, 1]
        ])

        faces_p = [0, 1, 2, 3]

        faces_n = [0, 0, 0, 0]
        faces_t = [0, 1, 2, 3]

        vtype = [('a_position', np.float32, 3),
                 ('a_texcoord', np.float32, 2),
                 ('a_normal', np.float32, 3)]

        itype = np.uint32

        vertices = np.zeros(4, vtype)
        vertices['a_position'] = points[faces_p]
        vertices['a_normal'] = n[faces_n]
        vertices['a_texcoord'] = t[faces_t]

        filled = np.resize(
            np.array([0, 1, 2, 0, 2, 3], dtype=itype), 6 * (2 * 3))
        filled += np.repeat(4 * np.arange(6, dtype=itype), 6)
        vertices = vertices.view(gloo.VertexBuffer)
        filled = filled.view(gloo.IndexBuffer)

        return vertices, filled

    def get_projection_matrix(self):
        C = np.append(self.rotation, np.vstack(self.translation), axis=1)
        return self.intrinsics.dot(C)

    def get_intrinsic_opengl(self, near_clip=1, far_clip=100):
        """
        :param near_clip: near_clip near clipping plane z-location, can be set arbitrarily > 0, controls the mapping of z-coordinates for OpenGL
        :param far_clip: far_clip  far clipping plane z-location, can be set arbitrarily > near_clip, controls the mapping of z-coordinate for OpenGL
        :return:
        """
        # alpha x-axis focal length, from camera intrinsic matrix
        alpha = self.intrinsics[0,0]
        # alpha y-axis focal length, from camera intrinsic matrix
        beta = self.intrinsics[1,1]
        # skew  x and y axis skew, from camera intrinsic matrix
        skew = self.intrinsics[0,1]
        # u0 image origin x-coordinate, from camera intrinsic matrix
        u0 = self.intrinsics[0,2]
        # v0 image origin y-coordinate, from camera intrinsic matrix
        v0 = self.intrinsics[1,2]


        # These parameters define the final viewport that is rendered into by the camera.
        L = 0
        R = self.nx
        B = 0
        T = self.ny
        viewport = np.array([L, B, R-L, T-B])

        """
        construct an orthographic matrix which maps from projected
        coordinates to normalized device coordinates in the range
        [-1, 1].  OpenGL then maps coordinates in NDC to the current
        """
        ortho = np.zeros((4, 4))
        ortho[0, 0] = 2.0 / (R - L)
        ortho[0, 3] = -(R + L) / (R - L)
        ortho[1, 1] = 2.0 / (T - B)
        ortho[1, 3] = -(T + B) / (T - B)
        ortho[2, 2] = -2.0 / (far_clip - near_clip)
        ortho[2, 3] = -(far_clip + near_clip) / (far_clip - near_clip)
        ortho[3, 3] = 1.0

        """
        construct a projection matrix, this is identical to the
        projection matrix computed for the intrinsic, except an
        additional row is inserted to map the z-coordinate to OpenGL.
        """
        tproj = np.reshape([
            alpha,  skew,                      -u0,                   0,
            0,      beta,                      -v0,                   0,
            0,      0,       (near_clip+far_clip), near_clip*far_clip,
            0,      0,                        -1.0,                   0
        ], (4, 4))

        """
        resulting OpenGL frustum is the product of the orthographic
        mapping to normalized device coordinates and the augmented
        camera intrinsic matrix
        """
        frustum = ortho.dot(tproj)
        return frustum

    def get_extrinsic_opengl(self):
        C = np.append(self.rotation, np.vstack(self.translation), axis=1)
        return np.vstack([C, [0, 0, 0, 1]])

    def projection_opengl(self, near_clip=1, far_clip=100):
        return self.get_intrinsic_opengl(near_clip, far_clip).dot(self.get_extrinsic_opengl())





camera_0 = Camera(
    intrinsics=np.array([
        1918.270000, 2.489820, 494.085000,
        0.000000, 1922.580000, 447.736000,
        0.000000, 0.000000, 1.000000
    ]),
    rotation=np.array([
        0.949462, 0.046934, 0.310324,
        -0.042337, 0.998867, -0.021532,
        -0.310985, 0.007308, 0.950373,
    ]),
    translation=np.array([-15.094651, 0.189829, 1.383263])
)

camera_1 = Camera(
    intrinsics=np.array([
        1913.690000, -0.143610, 533.307000,
        0.000000, 1918.170000, 398.171000,
        0.000000, 0.000000, 1.000000
    ]),
    rotation=np.array([
        0.972850, 0.010365, 0.231187,
        -0.012981, 0.999864, 0.009794,
        -0.231056, -0.012528, 0.972852,
    ]),
    translation=np.array([-11.589320, -0.355771, 1.045534])
)

camera_2 = Camera(
    intrinsics=np.array([
        1914.070000, 0.343703, 564.645000,
        0.000000, 1918.500000, 428.422000,
        0.000000, 0.000000, 1.000000
    ]),
    rotation=np.array([
        0.989230, 0.003946, 0.146295,
        -0.004391, 0.999983, 0.002724,
        -0.146283, -0.003337, 0.989230,
    ]),
    translation=np.array([-7.784865, -0.431597, 1.392058])
)

class CameraLoader(object):

    def __init__(self):
        self.selectedCamera = CameraRigType.Left
        self.c_left = camera_0
        self.c_virtual = camera_1
        self.c_right = camera_2

        # Save distance from virtual to cameras
        v_left = self.c_virtual.translation - self.c_left.translation
        v_right = self.c_virtual.translation - self.c_right.translation

        print "Left distance vector:"
        print v_left
        print "Right distance vector:"
        print v_right

        # Make virtual camera origin of coords
        self.c_virtual.translation = np.array([0, 0, 0])
        self.c_left.translation = v_left
        self.c_right.translation = v_right

        # Align rotation of virtual camera with axis
        rotation = self.c_virtual.rotationAxis
        print "Virtual camera rotation [X,Y,Z] (radians):"
        print rotation

    def set_camera(self, cam):
        print("Switch to camera (%s)" % cam)
        self.camera = cam

    def get_camera(self, cameraType, near_clip=1, far_clip=100):
        if cameraType is CameraRigType.Left:
            return self.c_left.projection_opengl(near_clip, far_clip)
        elif cameraType is CameraRigType.Virtual:
            return self.c_virtual.projection_opengl(near_clip, far_clip)
        elif cameraType is CameraRigType.Right:
            return self.c_right.projection_opengl(near_clip, far_clip)

        raise AttributeError("Camera not found!")

    def get_virtual_rotation(self):
        return camera_1.rotation

    def get_virtual_translation(self):
        return camera_1.translation
