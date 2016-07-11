from keras.models import Model

from shared_model import depth_concat_per_plane, input_size

# COLOR TOWER MODEL
def color_tower_model(input_organizer):

    output_layers = []

    shared_model = depth_concat_per_plane('color')

    for input in input_organizer.planes():
        output = shared_model(input)
        output_layers.append(output)

    return output_layers