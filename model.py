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

        net = tf.layers.batch_normalization(net, training=inputs.training)

        # conv layers
        net = ti.layers.conv2d_batch_norm(net, 32, [5, 5], activation=tf.nn.elu, name="elu_1", padding="same", bn_kwargs=dict(training=inputs.training))


        net = ti.layers.conv2d_batch_norm(net, 32, [3, 3], activation=tf.nn.elu, name="elu_2", padding="same", bn_kwargs=dict(training=inputs.training))
        net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, name="max_pool_1", padding="same")


        net = ti.layers.conv2d_batch_norm(net, 64, [3, 3], activation=tf.nn.elu, name="elu_3", padding="same", bn_kwargs=dict(training=inputs.training))
        net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, name="max_pool_2", padding="same")

        net = ti.layers.conv2d_batch_norm(net, 64, [3, 3], activation=tf.nn.elu, name="elu_4", padding="same", bn_kwargs=dict(training=inputs.training))

        # flatten
        net = tf.contrib.layers.flatten(net)
        net = tf.nn.dropout(net, self.inputs.keep_prob)
        # dense layers
        net = ti.layers.dense_batch_norm(net, 2048, activation=tf.nn.elu, name="dense_1", bn_kwargs=dict(training=inputs.training))
        net = tf.nn.dropout(net, self.inputs.keep_prob)

        net = ti.layers.dense_batch_norm(net, 512, activation=tf.nn.elu, name="dense_2", bn_kwargs=dict(training=inputs.training))

        # output layer
        return tf.layers.dense(net, self.n_classes)

    def get_summaries(self, inputs):
        return [
            tf.summary.scalar("learning_rate", self.learning_rate)
        ]
