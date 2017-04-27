from tfinterface.supervised import SupervisedModel
import tensorflow as tf

class Model(SupervisedModel):

    def __init__(self, n_classes, *args, **kwargs):
        kwargs["loss"] = "softmax"
        self.n_classes = n_classes

        super(Model, self).__init__(*args, **kwargs)


    def _build(self):
        # one hot labels
        self.labels = tf.one_hot(self.inputs.labels, self.n_classes)

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
        self.logits = tf.layers.dense(net, self.n_classes)
        self.predictions = tf.nn.softmax(self.logits)
