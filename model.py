from tfinterface.supervised import SoftmaxClassifier
import tensorflow as tf

class Model(SoftmaxClassifier):

    def __init__(self, n_classes, *args, **kwargs):
        self.n_classes = n_classes

        super(Model, self).__init__(*args, **kwargs)


    def get_labels(self):
        # one hot labels
        return tf.one_hot(self.inputs.labels, self.n_classes)

    def get_logits(self):
        # cast
        net = tf.cast(self.inputs.features, tf.float32, "cast")

        # conv layers
        net = tf.layers.conv2d(net, 16, [3, 3], activation=tf.nn.elu, name="elu_1")
        net = tf.layers.max_pooling2d(net, pool_size=2, strides=1)
        net = tf.layers.conv2d(net, 32, [3, 3], activation=tf.nn.elu, name="elu_2")

        # flatten
        net = tf.contrib.layers.flatten(net)

        # dense layers
        net = tf.layers.dense(net, 256, activation=tf.nn.elu)

        # output layer
        return tf.layers.dense(net, self.n_classes)
