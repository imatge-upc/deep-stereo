from keras.layers import Merge, Input, merge
from deep_stereo.color_tower import color_tower_model
from deep_stereo.select_tower import select_tower_model
from keras.models import Model
from shared_model import input_size
from keras import backend as K
from keras.optimizers import Adagrad



class InputOrganizer(object):

    def __init__(self, num_planes=96):
        self.input_layers = list()
        self.num_planes = num_planes
        for p in range(num_planes):
            name = "plane_%s" % p
            input_plane = input_size([30, 30], name) \
                                   + input_size([18, 18], name) \
                                   + input_size([12, 12], name) \
                                   + input_size([10, 10], name)
            self.input_layers.append(input_plane)

    def get_input_for_plane(self, plane_num):

        assert plane_num <= self.num_planes
        return self.input_layers[plane_num]

    def get_num_planes(self):
        return self.num_planes

    def planes(self):
        for inputs in self.input_layers:
            yield inputs

    def get_all_inputs(self):

        total_inputs = []
        for inputs in self.input_layers:
            total_inputs += inputs
        return total_inputs

# Define array of input layers for all defined planes
PLANES_DEPTH = 96
input_organizer = InputOrganizer(PLANES_DEPTH)

# Color Tower Model
color_outputs = color_tower_model(input_organizer)
#plot(color_model, to_file='color_tower.png', show_shapes=True)

# Select Tower Model
select_outputs = select_tower_model(input_organizer)
#plot(select_model, to_file='select_tower.png', show_shapes=True)

def dotpMerge2(inputs):
    """
    Reimplement Dotp merge in a more pythonic way
    :param inputs:
    :return:
    """
    plane_sweep_color = inputs[0:96]
    depth_probabilities = inputs[96]

    products = map(lambda (i, plane): K.dot(plane, depth_probabilities[i]), enumerate(plane_sweep_color))

    return sum(products)


def dotpMerge(inputs):
    """
    Expected to work dotp merge. First try
    :param inputs:
    :return:
    """
    plane_sweep_color = inputs[0:96]
    depth_probabilities = inputs[96]

    r_map = []
    g_map = []
    b_map = []
    for i in range(0, len(plane_sweep_color)):
        plane = plane_sweep_color[i]
        # Multiply R,G,B
        r_map.append(K.dot(plane[0], depth_probabilities[0]))
        g_map.append(K.dot(plane[1], depth_probabilities[1]))
        b_map.append(K.dot(plane[2], depth_probabilities[2]))

    final_r = r_map[0]
    final_g = g_map[0]
    final_b = b_map[0]
    for i in range(1, len(r_map)):
        final_r += r_map[i]
        final_g += g_map[i]
        final_b += g_map[i]

    return K.concatenate([final_r, final_g, final_b], axis=1)

# color_outputs has 96 RGB outputs and select_ouputs is a layer with depth 96, to merge a list a and
# a layer, do a concat on the list and merge with a special dotpMerge function
tomerge = color_outputs + [select_outputs]
out = merge(tomerge, output_shape=(None, 8, 8, 3), mode=dotpMerge2)


# Create the model
all_inputs = input_organizer.get_all_inputs()
model = Model(input=all_inputs, output=out)
model.compile(optimizer=Adagrad(lr=0.0001),
              loss='mean_squared_error',
              metrics=['accuracy'])

# The data input specs for 1 step are:
#   - 4 Cameras
#   - 96 Depth planes per camera
#   - 4 different resolution patches per camera depth plane starting from (30x30)
#   TOTAL: 4 * 96 * 4 = 1536 images (384 images per camera)

# Fits the data
#model.fit(data, labels, nb_epoch=10)  # starts training


