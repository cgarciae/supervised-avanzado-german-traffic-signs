from tfinterface.supervised import SoftmaxClassifier
import tensorflow as tf
import tfinterface as ti


class Model(SoftmaxClassifier):

    def __init__(self, n_classes, *args, **kwargs):
        self.n_classes = n_classes

        self._initial_learning_rate = kwargs.pop("initial_learning_rate", 0.001)
        self._decay_steps = kwargs.pop("decay_steps", 100)
        self._decay_rate = kwargs.pop("decay_rate", 0.96)
        self._staircase = kwargs.pop("staircase", True)
        self._rotation_angle = kwargs.pop("rotation_angle", 15.0)

        # model
        self._activation = kwargs.pop("activation", tf.nn.elu)
        self._dropout_rate = kwargs.pop("dropout_rate", 0.2)

        # densenet
        self._growth_rate = kwargs.pop("growth_rate", 12)
        self._compression = kwargs.pop("compression", 0.5)
        self._bottleneck = kwargs.pop("bottleneck", 4 * self._growth_rate)
        self._depth = kwargs.pop("growth_rate", 100)
        self._n_layers = (
            self._depth / 8 if self._bottleneck else
            self._depth / 4
        )

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
        # net = tf.layers.batch_normalization(net, training=inputs.training); print("Batch Norm: {}".format(net))

        # big kernel
        net = ti.layers.conv2d_batch_norm(net, 2 * self._growth_rate, [7, 7], activation=self._activation, padding='same', batch_norm=dict(training=inputs.training))
        print("Batch Norm Layer 24, 7x7: {}".format(net))

        # dense 1
        n_layers = 6
        net = ti.layers.conv2d_dense_block(
            net, self._growth_rate, self._n_layers, bottleneck = self._bottleneck, compression=self._compression,
            activation=self._activation, padding="same",
            dropout = dict(rate = self._dropout_rate),
            batch_norm = dict(training = inputs.training)
        ); print("DenseBlock(growth_rate={}, layers={}, bottleneck={}, compression={}): {}".format(
            self._growth_rate, n_layers, self._bottleneck, self._compression, net))
        # net = tf.layers.average_pooling2d(net, [2, 2], strides=2); print("Average Pooling 2x2".format(net))


        # dense 2
        n_layers = 12
        net = ti.layers.conv2d_dense_block(
            net, self._growth_rate, self._n_layers, bottleneck = self._bottleneck, compression=self._compression,
            activation=self._activation, padding="same",
            dropout = dict(rate = self._dropout_rate),
            batch_norm = dict(training = inputs.training)
        ); print("DenseBlock(growth_rate={}, layers={}, bottleneck={}, compression={}): {}".format(
            self._growth_rate, n_layers, self._bottleneck, self._compression, net))
        net = tf.layers.average_pooling2d(net, [2, 2], strides=2); print("Average Pooling 2x2".format(net))

        # dense 3
        n_layers = 24
        net = ti.layers.conv2d_dense_block(
            net, self._growth_rate, self._n_layers, bottleneck = self._bottleneck, compression=self._compression,
            activation=self._activation, padding="same",
            dropout = dict(rate = self._dropout_rate),
            batch_norm = dict(training = inputs.training)
        ); print("DenseBlock(growth_rate={}, layers={}, bottleneck={}, compression={}): {}".format(
            self._growth_rate, n_layers, self._bottleneck, self._compression, net))
        net = tf.layers.average_pooling2d(net, [2, 2], strides=2); print("Average Pooling 2x2".format(net))

        # dense 4
        n_layers = 16
        net = ti.layers.conv2d_dense_block(
            net, self._growth_rate, self._n_layers, bottleneck = self._bottleneck, compression=self._compression,
            activation=self._activation, padding="same",
            dropout = dict(rate = self._dropout_rate),
            batch_norm = dict(training = inputs.training)
        ); print("DenseBlock(growth_rate={}, layers={}, bottleneck={}, compression={}): {}".format(
            self._growth_rate, n_layers, self._bottleneck, self._compression, net))
        # net = tf.layers.average_pooling2d(net, [2, 2], strides=2); print("Average Pooling 2x2".format(net))

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
