from dataget import data # <== dataget
import tensorflow as tf
import cytoolz as cz
from phi.api import *
from model import Model
import numpy as np
import random
from name import network_name

graph = tf.Graph()
sess = tf.Session(graph=graph)

# inputs
inputs = dict(
    features = dict(shape = (None, 32, 32, 3)),
    labels = dict(shape = (None,), dtype = tf.uint8)
)

# create model template
template = Model(
    n_classes = 43,
    name = network_name,
    graph = graph,
    sess = sess,
    # seed = seed,
    optimizer = tf.train.AdamOptimizer,

)

# model
model = template(inputs)

with graph.as_default():
    print("")
    print("##########################################################")
    print("Number of Weights = {:,}".format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
    print("##########################################################")
