import numpy as np

class YUVReader(object):

    def __init__(self, file, size):
        self.width = size[0]
        self.height = size[1]

        self.fd = open(file, 'rb')

        self.plane_size = self.width * self.height
        bitsPerPixel = 12
        self.frame_size = self.plane_size * bitsPerPixel / 8

    def getRGBFrame(self, frame):

        self.fd.seek(frame * self.frame_size)

        #y = self.fd.read(self.plane_size)
        #v = self.fd.read(self.plane_size / 4)
        #u = self.fd.read(self.plane_size / 4)

        # Load the Y (luminance) data from the stream
        Y = np.fromfile(self.fd, dtype=np.uint8, count=self.width * self.height). \
            reshape((self.height, self.width))
        # Load the UV (chrominance) data from the stream, and double its size
        U = np.fromfile(self.fd, dtype=np.uint8, count=(self.width // 2) * (self.height // 2)). \
            reshape((self.height // 2, self.width // 2)). \
            repeat(2, axis=0).repeat(2, axis=1)
        V = np.fromfile(self.fd, dtype=np.uint8, count=(self.width // 2) * (self.height // 2)). \
            reshape((self.height // 2, self.width // 2)). \
            repeat(2, axis=0).repeat(2, axis=1)

        # Stack the YUV channels together, crop the actual resolution, convert to
        # floating point for later calculations, and apply the standard biases
        YUV = np.dstack((Y, U, V))[:self.height, :self.width, :].astype(np.float)
        YUV[:, :, 0] = YUV[:, :, 0] - 16  # Offset Y by 16
        YUV[:, :, 1:] = YUV[:, :, 1:] - 128  # Offset UV by 128
        # YUV conversion matrix from ITU-R BT.601 version (SDTV)
        #              Y       U       V
        M = np.array([[1.164, 0.000, 1.596],  # R
                      [1.164, -0.392, -0.813],  # G
                      [1.164, 2.017, 0.000]])  # B
        # Take the dot product with the matrix to produce RGB output, clamp the
        # results to byte range and convert to bytes
        RGB = YUV.dot(M.T).clip(0, 255).astype(np.uint8)

        return RGB