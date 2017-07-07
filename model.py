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
        self._rotation_angle = kwargs.pop("rotation_angle", 15.0)

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
        net = tf.layers.batch_normalization(net, training=inputs.training)

        # big kernel
        net = ti.layers.conv2d_batch_norm(net, 96, [7, 7], activation=tf.nn.elu, padding='same', batch_norm=dict(training=inputs.training))

        # fire
        net = ti.layers.fire_batch_norm(net, 16, 64, 64, activation=tf.nn.elu, padding='same', batch_norm=dict(training=inputs.training)) #fire2
        net = net + ti.layers.fire_batch_norm(net, 16, 64, 64, activation=tf.nn.elu, padding='same', batch_norm=dict(training=inputs.training)) #fire3
        net = ti.layers.fire_batch_norm(net, 32, 128, 128, activation=tf.nn.elu, padding='same', batch_norm=dict(training=inputs.training)) #fire4

        # max pooling
        net = tf.layers.max_pooling2d(net, [3, 3], strides=2, padding='same')

        net = net + ti.layers.fire_batch_norm(net, 32, 128, 128, activation=tf.nn.elu, padding='same', batch_norm=dict(training=inputs.training)) #fire5
        net = ti.layers.fire_batch_norm(net, 48, 192, 192, activation=tf.nn.elu, padding='same', batch_norm=dict(training=inputs.training)) #fire6
        net = net + ti.layers.fire_batch_norm(net, 48, 192, 192, activation=tf.nn.elu, padding='same', batch_norm=dict(training=inputs.training)) #fire7
        net = ti.layers.fire_batch_norm(net, 64, 256, 256, activation=tf.nn.elu, padding='same', batch_norm=dict(training=inputs.training)) #fire8

        # max pooling
        net = tf.layers.max_pooling2d(net, [3, 3], strides=2, padding='same')

        #fire + droput
        net = net + ti.layers.fire_batch_norm(net, 64, 256, 256, activation=tf.nn.elu, padding='same', batch_norm=dict(training=inputs.training)) #fire9
        net = tf.layers.dropout(net, rate=0.25, training=inputs.training)

        # reduce
        net = ti.layers.conv2d_batch_norm(net, self.n_classes, [1, 1], activation=tf.nn.elu, padding='same', batch_norm=dict(training=inputs.training)) #linear
        shape = net.get_shape()[1]
        net = tf.layers.average_pooling2d(net, [shape, shape], strides=1)

        # flatten
        net = tf.contrib.layers.flatten(net)
        # output layer
        return net

    def get_summaries(self, inputs):
        return [
            tf.summary.scalar("learning_rate", self.learning_rate)
        ]

    def random_rotate_images(self, net):
        return tf.where(
            self.inputs.training,
            tf.contrib.image.rotate(
                net,
                tf.random_uniform(tf.shape(net)[:1], minval = -self._rotation_angle, maxval = self._rotation_angle)
            ),
            net
        )

    def get_update(self, *args, **kwargs):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            return super(Model, self).get_update(*args, **kwargs)
