from scipy.misc import imread
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dataget import data # <== dataget
import tfinterface as ti
import tensorflow as tf


from tfinterface.supervised import SupervisedModel
from tfinterface.supervised import SupervisedTrainer

###########

dataset = data("german-traffic-signs").get()

###########
print("loading dataset")

# obtener todas las imagenes (lento)
features, labels = dataset.training_set.arrays()

print("Features shape: {} \nLabels shape: {}".format(features.shape, labels.shape))


###########

from tfinterface.supervised import SupervisedInputs

def random_shuffle_fns(tensors_dict, **kwargs):
    self = random_shuffle_fns
    self.shuffled_tensors = None

    def shuffle_tensors():
        if self.shuffled_tensors is None:

            self.shuffled_tensors = tf.train.shuffle_batch(
                tensors_dict,
                **kwargs
            )

    def get_fn(name):
        def tensor_fn():
            shuffle_tensors()
            return self.shuffled_tensors[name]

        return tensor_fn

    return ({
        name : get_fn(name)
        for name, _ in tensors_dict.items()
    })



#########

class Model(SupervisedModel):

    def _build(self, inputs):
        self.inputs = inputs

        n_classes = 43

        net = tf.cast(inputs.features, tf.float32, "cast")
        net = tf.layers.conv2d(net, 16, [3, 3], activation=tf.nn.elu, name="elu_1")
        net = tf.layers.max_pooling2d(net, 2, 1)
        net = tf.layers.conv2d(net, 32, [3, 3], activation=tf.nn.elu, name="elu_2")

        net = tf.contrib.layers.flatten(net)

        net = tf.layers.dense(net, 256, activation=tf.nn.elu)

        self.logits = tf.layers.dense(net, n_classes)
        self.predictions = tf.nn.softmax(self.logits)
        self.labels = tf.one_hot(self.inputs.labels, n_classes)


inputs_fn = SupervisedInputs("inputs", **random_shuffle_fns(dict(
    features = features,
    labels = labels
),
    batch_size=32,
    num_threads=4,
    capacity=50000,
    min_after_dequeue=10000,
    enqueue_many=True
))
model_fn = Model("conv_net")
trainer_fn = SupervisedTrainer("trainer", loss="softmax")

graph = tf.Graph()
sess = tf.Session(graph=graph)

with graph.as_default(), sess.as_default():
    inputs = inputs_fn()
    model = model_fn(inputs)
    trainer = trainer_fn(model)


model.initialize()
trainer.fit()
