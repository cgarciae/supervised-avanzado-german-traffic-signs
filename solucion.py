from scipy.misc import imread
import os
import pandas as pd
import numpy as np
from dataget import data # <== dataget
import tfinterface as ti
import tensorflow as tf
import cytoolz as cz
from phi.api import *
import sys

from tfinterface.supervised import SupervisedModel
from tfinterface.supervised import SupervisedTrainer

###########

dataset = data("german-traffic-signs").get()

###########
print("loading dataset")

# obtener todas las imagenes (lento)
data_generator = dataset.training_set.random_batch_arrays_generator(32)
data_generator = cz.map(Dict(features = P[0], labels = P[1]), data_generator)

# print("Features shape: {} \nLabels shape: {}".format(features.shape, labels.shape))


###########

from tfinterface.supervised import SupervisedInputs


#########

class Model(SupervisedModel):

    def _build(self):

        n_classes = 43

        net = tf.cast(self.inputs.features, tf.float32, "cast")

        net = tf.layers.conv2d(net, 16, [3, 3], activation=tf.nn.elu, name="elu_1")
        net = tf.layers.max_pooling2d(net, pool_size=2, strides=1)
        net = tf.layers.conv2d(net, 32, [3, 3], activation=tf.nn.elu, name="elu_2")

        net = tf.contrib.layers.flatten(net)

        net = tf.layers.dense(net, 256, activation=tf.nn.elu)

        self.logits = tf.layers.dense(net, n_classes)
        self.predictions = tf.nn.softmax(self.logits)
        self.labels = tf.one_hot(self.inputs.labels, n_classes)



mode = sys.argv[1] if len(sys.argv) > 1 else "train"

# from tfinterface.utils import shuffle_batch_tensor_fns
model_fn = Model("conv_net", loss="softmax", graph=tf.Graph(), model_path = os.path.join(os.getcwd(), "model")
)

if mode == "train":

    # build
    model = model_fn(inputs=dict(
        features = dict(shape = (None, 32, 32, 3)),
        labels = dict(shape = (None,), dtype = tf.uint8)
    ))

    # init
    # tf.train.start_queue_runners(sess=sess)
    model.initialize()

    # fit
    model.fit(data_generator=data_generator, epochs=1500)

    model.save()

elif mode == "test":
    ### test
    # features_test, labels_test = dataset.test_set.arrays()
    features_test, labels_test = next(dataset.test_set.random_batch_arrays_generator(2000))


    model_test = model_fn(inputs=dict(
        features = features_test,
        labels = labels_test
    ))
    import os
    model_test.initialize(restore = True)

    test_score = model_test.score()
    print("test score: {}".format(test_score))
