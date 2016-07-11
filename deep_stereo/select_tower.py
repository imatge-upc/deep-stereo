from keras.layers import Convolution2D, merge, Activation, Flatten, Reshape
from keras.models import Model
from keras.utils.visualize_util import plot

from shared_model import depth_concat_per_plane, input_size


# SELECT TOWER MODEL
def select_tower_model(input_organizer):

    output_layers = []
    shared_model = depth_concat_per_plane('select')

    for input in input_organizer.planes():
        output = shared_model(input)
        output_layers.append(output)

    merged_out = merge(output_layers, mode='concat', concat_axis=1)
    x = Convolution2D(480, 3, 3, activation='relu')(merged_out)
    x = Convolution2D(480, 3, 3, activation='relu')(x)
    x = Convolution2D(96, 3, 3, activation='relu')(x)
    x = Activation('tanh')(x)

    # Reshape tensor to apply softmax per pixellayer
    x = Reshape((96, 64))(x)
    x = Activation('softmax')(x)
    output = Reshape((96, 8, 8))(x)

    return output
