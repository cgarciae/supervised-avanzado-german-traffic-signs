from scipy.misc import imread
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dataget import data # <== dataget
import tfinterface as ti
import tensorflow as tf
import cytoolz as cz
from phi.api import *

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

def random_shuffle_fns(tensors_dict, **kwargs):
    self = random_shuffle_fns

    self.tensors_dict = tensors_dict
    self.shuffled_tensors = None


    def shuffle_tensors():
        if self.shuffled_tensors is None:
            self.tensors_dict = ({key: value for key, value in self.tensors_dict.items() })

            self.shuffled_tensors = tf.train.shuffle_batch(
                self.tensors_dict,
                **kwargs
            )

    def get_fn(name):
        def tensor_fn():
            shuffle_tensors()
            return self.shuffled_tensors[name]

        return tensor_fn

    return ({
        name : get_fn(name)
        for name in self.tensors_dict
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



from tfinterface.utils import shuffle_batch_tensor_fns

graph = tf.Graph()
sess = tf.Session(graph=graph)

with graph.as_default(), sess.as_default():
    # inputs_fn = SupervisedInputs("inputs", **shuffle_batch_tensor_fns(dict(
    #     features = tf.convert_to_tensor(features, dtype=tf.float32),
    #     labels = tf.convert_to_tensor(labels, dtype=tf.uint8)
    # ),
    #     batch_size=32,
    #     capacity=50000,
    #     enqueue_many=True,
    #     min_after_dequeue=10,
    #     num_threads=4
    # ))
    inputs_fn = SupervisedInputs("inputs",
        features = dict(shape = (None, 32, 32, 3)),
        labels = dict(shape = (None,), dtype = tf.uint8)
    )
    model_fn = Model("conv_net")
    trainer_fn = SupervisedTrainer("trainer", loss="softmax")


inputs = inputs_fn()
model = model_fn(inputs)
trainer = trainer_fn(model)

tf.train.start_queue_runners(sess=sess)
model.initialize()


trainer.fit(data_generator=data_generator)
