import numpy as np
import tensorflow as tf

from face_detector.architecture.layer_factory import LayerFactory
from face_detector.architecture.network import Network

class PNet(Network):
    """
    Network to propose areas with faces.
    """
    def _config(self):
        layer_factory = LayerFactory(self)

        layer_factory.new_feed(name='data', layer_shape=(None, None, None, 3))
        layer_factory.new_conv(name='conv1', kernel_size=(3, 3), channels_output=10, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu1')
        layer_factory.new_max_pool(name='pool1', kernel_size=(2, 2), stride_size=(2, 2))
        layer_factory.new_conv(name='conv2', kernel_size=(3, 3), channels_output=16, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu2')
        layer_factory.new_conv(name='conv3', kernel_size=(3, 3), channels_output=32, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu3')
        layer_factory.new_conv(name='conv4-1', kernel_size=(1, 1), channels_output=2, stride_size=(1, 1), relu=False)
        layer_factory.new_softmax(name='prob1', axis=3)

        layer_factory.new_conv(name='conv4-2', kernel_size=(1, 1), channels_output=4, stride_size=(1, 1),
                               input_layer_name='prelu3', relu=False)

    def _feed(self, image):
        return self._session.run(['pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'], feed_dict={'pnet/input:0': image})


class RNet(Network):
    """
    Network to refine the areas proposed by PNet
    """

    def _config(self):

        layer_factory = LayerFactory(self)

        layer_factory.new_feed(name='data', layer_shape=(None, 24, 24, 3))
        layer_factory.new_conv(name='conv1', kernel_size=(3, 3), channels_output=28, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu1')
        layer_factory.new_max_pool(name='pool1', kernel_size=(3, 3), stride_size=(2, 2))
        layer_factory.new_conv(name='conv2', kernel_size=(3, 3), channels_output=48, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu2')
        layer_factory.new_max_pool(name='pool2', kernel_size=(3, 3), stride_size=(2, 2), padding='VALID')
        layer_factory.new_conv(name='conv3', kernel_size=(2, 2), channels_output=64, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu3')
        layer_factory.new_fully_connected(name='fc1', output_count=128, relu=False)  # shouldn't the name be "fc1"?
        layer_factory.new_prelu(name='prelu4')
        layer_factory.new_fully_connected(name='fc2-1', output_count=2, relu=False)   # shouldn't the name be "fc2-1"?
        layer_factory.new_softmax(name='prob1', axis=1)

        layer_factory.new_fully_connected(name='fc2-2', output_count=4, relu=False, input_layer_name='prelu4')

    def _feed(self, image):
        return self._session.run(['rnet/fc2-2/fc2-2:0', 'rnet/prob1:0'], feed_dict={'rnet/input:0': image})


class ONet(Network):
    """
    Network to retrieve the keypoints
    """
    def _config(self):
        layer_factory = LayerFactory(self)

        layer_factory.new_feed(name='data', layer_shape=(None, 48, 48, 3))
        layer_factory.new_conv(name='conv1', kernel_size=(3, 3), channels_output=32, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu1')
        layer_factory.new_max_pool(name='pool1', kernel_size=(3, 3), stride_size=(2, 2))
        layer_factory.new_conv(name='conv2', kernel_size=(3, 3), channels_output=64, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu2')
        layer_factory.new_max_pool(name='pool2', kernel_size=(3, 3), stride_size=(2, 2), padding='VALID')
        layer_factory.new_conv(name='conv3', kernel_size=(3, 3), channels_output=64, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu3')
        layer_factory.new_max_pool(name='pool3', kernel_size=(2, 2), stride_size=(2, 2))
        layer_factory.new_conv(name='conv4', kernel_size=(2, 2), channels_output=128, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu4')
        layer_factory.new_fully_connected(name='fc1', output_count=256, relu=False)
        layer_factory.new_prelu(name='prelu5')
        layer_factory.new_fully_connected(name='fc2-1', output_count=2, relu=False)
        layer_factory.new_softmax(name='prob1', axis=1)

        layer_factory.new_fully_connected(name='fc2-2', output_count=4, relu=False, input_layer_name='prelu5')

        layer_factory.new_fully_connected(name='fc2-3', output_count=10, relu=False, input_layer_name='prelu5')

    def _feed(self, image):
        return self._session.run(['onet/fc2-2/fc2-2:0', 'onet/fc2-3/fc2-3:0', 'onet/prob1:0'],
                                 feed_dict={'onet/input:0': image})


class Architecture():
    def __init__(self):
        '''
        Build Multi Task CNN Architecture for face detection
        '''
        self.__graph = tf.Graph()
        self._config = tf.compat.v1.ConfigProto(log_device_placement=False)
        self._config.gpu_options.allow_growth = True

    def build(self, weights_file):
        with self.__graph.as_default():
            self.__session = tf.compat.v1.Session(config=self._config, graph=self.__graph)

            weights = np.load(weights_file, allow_pickle=True).item()
            self.__pnet = PNet(self.__session, False)
            self.__pnet.set_weights(weights['PNet'])

            self.__rnet = RNet(self.__session, False)
            self.__rnet.set_weights(weights['RNet'])

            self.__onet = ONet(self.__session, False)
            self.__onet.set_weights(weights['ONet'])

        return self.__pnet, self.__rnet, self.__onet

    def closeSession(self):
        self.__session.close()