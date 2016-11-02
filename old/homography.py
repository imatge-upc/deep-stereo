import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

img = cv2.imread('beach.jpg')
rows, cols, ch = img.shape

# CAM0 rotation
R = np.array([[0.949462, 0.046934, 0.310324],
              [-0.042337, 0.998867, -0.021532],
              [-0.310985, 0.007308, 0.950373]])

# Sweep plane
z_0 = np.array([0, 0, 100])

# Translate vector
t = np.array([0, 0, 0])

# Camera Intrinsic Parameters (CAM0)
Intrinsics_CAM0 = np.array([[1918.270000,   2.489820,       494.085000],
                            [0,             1922.580000,    447.736000],
                            [0,             0,              1]])


nx = 1024
ny = 768


fc_left = np.array([Intrinsics_CAM0[0, 0], Intrinsics_CAM0[1, 1]])
print "Focal length: "
print fc_left

cc_left = np.array([Intrinsics_CAM0[0, 2], Intrinsics_CAM0[1, 2]])
print "Principal Point Offset: "
print cc_left

# FC = focal lenght
a = np.array([  [1/fc_left[0], 0, 0],
                [0,1/fc_left[1],0],
                [0,0,1]])
# CC = principal point
b = np.array([  [1,0,-cc_left[0]],
                [0,1,-cc_left[1]],
                [0,0,1]])

c = np.array([[0, nx-1, nx-1,    0, 0],
              [0,    0, ny-1, ny-1, 0],
              [1,    1,    1,    1, 1]])

I_CAM = reduce(np.dot, [a, b, c])

BASE_left = np.array([[0,1,0,0,0,0],[0,0,0,1,0,0],[0,0,0,0,0,1]])

q = np.vstack(BASE_left[:, 0]).dot(np.ones((1, 5)))
I_CAM = np.reshape(np.concatenate([I_CAM, q, I_CAM], axis=1), (3, 15))

print I_CAM
print I_CAM.transpose()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(I_CAM[0,:], I_CAM[2,:], -I_CAM[1,:])
plt.show()

print "Script ends"