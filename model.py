from tfinterface.supervised import SupervisedModel
import tensorflow as tf

class Model(SupervisedModel):

    def __init__(self, n_classes, *args, **kwargs):
        kwargs["loss"] = "softmax"
        self.n_classes = n_classes

        self._initial_learning_rate = kwargs.pop("initial_learning_rate", 0.001)
        self._decay_steps = kwargs.pop("decay_steps", 200)
        self._decay_rate = kwargs.pop("decay_rate", 0.96)
        self._staircase = kwargs.pop("staircase", True)

        super(Model, self).__init__(*args, **kwargs)


    def _build(self):
        # one hot labels
        self.labels = tf.one_hot(self.inputs.labels, self.n_classes)

        # cast
        net = tf.cast(self.inputs.features, tf.float32, "cast")

        # conv layers
        net = tf.layers.conv2d(net, 16, [5, 5], activation=tf.nn.elu, name="elu_1", padding="same")

        net = tf.layers.conv2d(net, 32, [3, 3], activation=tf.nn.elu, name="elu_2", padding="same")
        net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, name="max_pool_1", padding="same")


        net = tf.layers.conv2d(net, 64, [3, 3], activation=tf.nn.elu, name="elu_3", padding="same")
        net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, name="max_pool_2", padding="same")

        net = tf.layers.conv2d(net, 64, [3, 3], activation=tf.nn.elu, name="elu_4", padding="same")

        # flatten
        net = tf.contrib.layers.flatten(net)

        # dense layers
        net = tf.layers.dense(net, 2048, activation=tf.nn.elu)
        net = tf.nn.dropout(net, self.inputs.keep_prob)

        net = tf.layers.dense(net, 512, activation=tf.nn.elu)

        # output layer
        self.logits = tf.layers.dense(net, self.n_classes)
        self.predictions = tf.nn.softmax(self.logits)


        self.learning_rate = tf.train.exponential_decay(
            self._initial_learning_rate,
            self.inputs.global_step,
            self._decay_steps,
            self._decay_rate,
            staircase = True
        )
