import matplotlib.pyplot as plt
import numpy as np
import math

def plot_camera(figure, camera, depth=1.0, rotate=True):
    points = np.array([
            (-1, 1, depth),
            (1, 1, depth),
            (1, -1, depth),
            (-1, -1, depth),
    ])
    #print points

    zDepth = 1
    p = zDepth * np.vstack([points, points[0, :]])
    #print p

    center = np.array([0, 0, 0])
    #print "Camera center: %s" % camera.translation

    p = np.reshape(np.hstack([p, np.vstack([center, center, center, center, center]), p]), (15, 3))

    #p = camera.intrinsics.dot(p.T).T
    #print p

    #rotation = rotateX(90)
    rotation = camera.getRotation()

    print "Camera translation -> %s" % camera.getTranslation()

    p = p + camera.getTranslation()

    if rotate:
        p = p.dot(rotation)

    #print p

    X = p[:, 0]
    Y = p[:, 1]
    Z = p[:, 2]

    ax = figure.gca(projection='3d')
    #ax.set_aspect('equal', 'datalim')
    ax.plot(X, Y, Z)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)


    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.scatter(X, Y, Z)


def plot_depth_plane(figure, camera, depth=5.0):
    points = np.array([
        (-1, 1, depth),
        (1, 1, depth),
        (1, -1, depth),
        (-1, -1, depth),
    ])
    p = np.reshape(points, (4, 3))

    p = p + camera.getTranslation()
    p = p.dot(camera.getRotation())

    ax = figure.gca(projection='3d')
    X = p[:, 0]
    Y = p[:, 1]
    Z = p[:, 2]
    ax.plot(X, Y, Z)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.scatter(X, Y, Z)