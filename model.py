from tfinterface.supervised import SoftmaxClassifier
import tensorflow as tf
import tfinterface as ti


class Model(SoftmaxClassifier):

    def __init__(self, n_classes, *args, **kwargs):
        self.n_classes = n_classes

        self._initial_learning_rate = kwargs.pop("initial_learning_rate", 0.001)
        self._decay_steps = kwargs.pop("decay_steps", 200)
        self._decay_rate = kwargs.pop("decay_rate", 0.96)
        self._staircase = kwargs.pop("staircase", True)

        super(Model, self).__init__(*args, **kwargs)

    def get_labels(self, inputs):
        # one hot labels
        return tf.one_hot(inputs.labels, self.n_classes)

    def get_learning_rate(self, inputs):
        return tf.train.exponential_decay(
            self._initial_learning_rate,
            self.inputs.global_step,
            self._decay_steps,
            self._decay_rate,
            staircase = True
        )

    def get_logits(self, inputs):

        # cast
        net = tf.cast(self.inputs.features, tf.float32, "cast")

        # big kernel
        net = tf.layers.conv2d(net, 96, [7, 7], activation=tf.nn.elu, padding='same')

        # fire
        net = ti.layers.fire(net, 16, 64, 64, activation=tf.nn.elu, padding='same') #fire2
        net = ti.layers.fire(net, 16, 64, 64, activation=tf.nn.elu, padding='same') #fire3
        net = ti.layers.fire(net, 32, 128, 128, activation=tf.nn.elu, padding='same') #fire4
        net = ti.layers.fire(net, 32, 128, 128, activation=tf.nn.elu, padding='same') #fire5
        net = ti.layers.fire(net, 48, 192, 192, activation=tf.nn.elu, padding='same') #fire6

        # max pooling
        net = tf.layers.max_pooling2d(net, [3, 3], strides=2, padding='same')

        # fire
        net = ti.layers.fire(net, 48, 192, 192, activation=tf.nn.elu, padding='same') #fire7
        net = ti.layers.fire(net, 64, 256, 256, activation=tf.nn.elu, padding='same') #fire8
        net = ti.layers.fire(net, 64, 256, 256, activation=tf.nn.elu, padding='same') #fire9

        # droput
        net = tf.layers.dropout(net, rate=0.25, training=inputs.training)

        # reduce
        net = tf.layers.conv2d(net, self.n_classes, [1, 1], padding='same') #linear
        net = tf.layers.average_pooling2d(net, [16, 16], strides=1)

        # flatten
        net = tf.contrib.layers.flatten(net)
        # output layer
        return net

    def get_summaries(self, inputs):
        return [
            tf.summary.scalar("learning_rate", self.learning_rate)
        ]
