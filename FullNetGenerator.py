from VggDNetGraphProvider import *
from keras.layers.core import Reshape
from keras.models import Model
from keras.applications import VGG16

from keras import layers


class FullNetGenerator(object):
    def __init__(self, weights_path):
        self.final_common_layer = 'conv13'
        self.weights_path = weights_path

    def create_full_net(self, score_branch=True, seg_branch=True):
        vgg_provider = VggDNetGraphProvider()
        net = vgg_provider.get_vgg_partial_graph(weights_path=self.weights_path, with_output=False)
        if score_branch:
            self.append_score_branch(net)
        if seg_branch:
            self.append_segmentation_branch(net)
        return net

    def create_full_model(self):
        vgg_model = VGG16(include_top=False)
        vgg_model.layers.pop()
        input_tensor = Input(shape=(224,224,3))
        vgg = vgg_model(input_tensor)
        score = self.score_branch(vgg)
        segmentation = self.segmentation_branch(vgg)
        model = Model(input_tensor, [score, segmentation])
        return model

    def append_score_branch(self, graph):
        graph.add_node(MaxPooling2D((2, 2), strides=(2, 2)), name='score_pool1', input=self.final_common_layer)
        graph.add_node(Flatten(), name='score_flat', input='score_pool1')
        graph.add_node(Dense(512, activation='relu'), name='score_dense1', input='score_flat')
        graph.add_node(Dropout(0.5), name='score_drop1', input='score_dense1')
        graph.add_node(Dense(1024, activation='relu'), name='score_dense2', input='score_drop1')
        graph.add_node(Dropout(0.5), name='score_drop2', input='score_dense2')
        graph.add_node(Dense(1), name='score_linear', input='score_drop2')
        graph.add_output(input='score_linear', name='score_output')

    def append_segmentation_branch(self, graph):
        graph.add_node(Convolution2D(512, 1, 1, activation='relu'), name='seg_conv1', input=self.final_common_layer)
        graph.add_node(Flatten(), name='seg_flat', input='seg_conv1')
        graph.add_node(Dense(512), name='seg_dense1', input='seg_flat')  # no activation here!
        graph.add_node(Dense(56*56), name='seg_dense2', input='seg_dense1')
        graph.add_node(Reshape(dims=(56, 56)), name='seg_reshape', input='seg_dense2')
        graph.add_output(input='seg_reshape', name='seg_output')

    def score_branch(self, x):
        x = layers.MaxPooling2D((2,2), strides=(2,2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512,activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1, name='score_linear')(x)
        return x

    def segmentation_branch(self, x):
        x = layers.Conv2D(512,1,activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512)(x)
        x = layers.Dense(56*56)(x)
        x = layers.Reshape(target_shape=(56,56))(x)
        return x


# usage-
# fng = FullNetGenerator('Resources/vgg16_graph_weights.h5')
# fn = fng.create_full_net()
