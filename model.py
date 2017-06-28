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

        print("###############################")
        print("# Model")
        print("###############################")
        # cast
        net = tf.cast(self.inputs.features, tf.float32, "cast"); print("Input: {}".format(net))
        net = tf.layers.batch_normalization(net, training=inputs.training); print("Batch Norm: {}".format(net))

        # big kernel
        net = ti.layers.conv2d_batch_norm(net, 96, [7, 7], activation=tf.nn.elu, padding='same', batch_norm=dict(training=inputs.training))
        print("Batch Norm Layer 96, 7x7: {}".format(net))

        # dense 1
        net = ti.layers.conv2d_densefire_block(
            net, bottleneck=48, growth_rate_1x1=6, growth_rate_3x3=6,
            n_layers=16, compression=0.8, activation=tf.nn.elu, padding="same",
            dropout = dict(rate = 0.2), batch_norm = dict(training = inputs.training)
        ); print("Dense Fire Block (48, 6, 6, 16): {}".format(net))
        net = tf.layers.average_pooling2d(net, [2, 2], strides=2); print("Average Pooling 2x2".format(net))

        # dense 2
        net = ti.layers.conv2d_densefire_block(
            net, bottleneck=48, growth_rate_1x1=6, growth_rate_3x3=6,
            n_layers=16, compression=None, activation=tf.nn.elu, padding="same",
            dropout = dict(rate = 0.2), batch_norm = dict(training = inputs.training)
        ); print("Dense Fire Block (48, 6, 6, 16): {}".format(net))

        # global average pooling
        shape = net.get_shape()[1]
        net = tf.layers.average_pooling2d(net, [shape, shape], strides=1); print("Global Average Pooling: {}".format(net))
        net = tf.contrib.layers.flatten(net); print("Flatten: {}".format(net))

        # dense
        net = ti.layers.dense_batch_norm(net, self.n_classes, batch_norm=dict(training=inputs.training)); print("Dense Batch Norm Layer 43: {}".format(net))

        print("###############################\n")

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
