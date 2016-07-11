from keras.layers import Merge, merge, Convolution2D, Input, Activation
from keras.layers.convolutional import UpSampling2D
from keras.models import Model

# Defines a set containing an input layer plus a ConvRelu
def input_size(size, name="input"):
    num_channels = 4
    input_1 = Input(shape=(num_channels, size[0], size[1]), name="%s_cam1_%s_%s" % (name, size[0], size[0]))
    input_2 = Input(shape=(num_channels, size[0], size[1]), name="%s_cam2_%s_%s" % (name, size[0], size[0]))
    input_3 = Input(shape=(num_channels, size[0], size[1]), name="%s_cam3_%s_%s" % (name, size[0], size[0]))
    input_4 = Input(shape=(num_channels, size[0], size[1]), name="%s_cam4_%s_%s" % (name, size[0], size[0]))
    return [input_1, input_2, input_3, input_4]


# Pre patch net
def patch_pipe(size_in, kernels, depths, upsampling=False):

    input = input_size(size_in)

    cv_1 = Convolution2D(64, 5, 5, activation='relu')(input[0])
    cv_2 = Convolution2D(64, 5, 5, activation='relu')(input[1])
    cv_3 = Convolution2D(64, 5, 5, activation='relu')(input[2])
    cv_4 = Convolution2D(64, 5, 5, activation='relu')(input[3])

    first_layer = merge([
        cv_1,
        cv_2,
        cv_3,
        cv_4
    ], mode='concat', concat_axis=1)

    # 5x5 conv with 96 output channels (same as input channels)
    x = Convolution2D(depths[0], kernels[0][0], kernels[0][1], activation='relu')(first_layer)
    x = Convolution2D(depths[1], kernels[1][0], kernels[1][1], activation='relu')(x)

    # This layer only applies on 2 small patches
    if len(kernels) == 3:
        x = Convolution2D(depths[2], kernels[2][0], kernels[2][1], activation='relu')(x)

    if upsampling:
        out = UpSampling2D(size=upsampling)(x)
    else:
        out = x

    return input, out


def depth_concat_per_plane(model_type):
    # Patch 30x30
    input_30_30, out_30_30 = patch_pipe(size_in=[30, 30],
                                                 kernels=[(3, 3), (5, 5), (5, 5)],
                                                 depths=[96, 48, 16],
                                                 )

    # Patch 18x18
    input_18_18, out_18_18 = patch_pipe(size_in=[18, 18],
                                                 kernels=[(3, 3), (3, 3), (3, 3)],
                                                 depths=[40, 40, 8],
                                                 upsampling=(2, 2),
                                                 )

    # Patch 12x12
    input_12_12, out_12_12 = patch_pipe(size_in=[12, 12],
                                                 kernels=[(3, 3), (3, 3)],
                                                 depths=[32, 8],
                                                 upsampling=(4, 4)
                                                 )

    # Patch 10x10
    input_10_10, out_10_10 = patch_pipe(size_in=[10, 10],
                                             kernels=[(3, 3), (3, 3)],
                                             depths=[32, 8],
                                             upsampling=(8, 8)
                                             )

    concatenated = merge([out_30_30, out_18_18, out_12_12, out_10_10], mode='concat', concat_axis=1)

    in_model = input_30_30 + input_18_18 + input_12_12 + input_10_10

    if model_type == 'color':
        x = Convolution2D(96, 3, 3, activation='relu')(concatenated)
        x = Convolution2D(32, 1, 1, activation='relu')(x)
        x = Convolution2D(32, 3, 3, activation='relu')(x)
        out_tower = Convolution2D(3, 5, 5, activation='relu')(x)
    elif model_type == 'select':
        x = Convolution2D(96, 1, 1, activation='relu')(concatenated)
        x = Convolution2D(32, 1, 1, activation='relu')(x)
        out_tower = Convolution2D(4, 3, 3, activation='relu')(x)

    else:
        raise NameError("invalid input mode")

    model = Model(input=in_model, output=out_tower)

    return model

