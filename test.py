from dataget import data # <== dataget
import tensorflow as tf
import cytoolz as cz
from phi.api import *
from model import Model
import numpy as np
import random

# seed: resultados repetibles
seed = 32
np.random.seed(seed=seed)
random.seed(seed)

# dataget
dataset = data("german-traffic-signs").get()

# obtener imagenes
print("loading data")
features_test, labels_test = next(dataset.test_set.random_batch_arrays_generator(2000))

# create model template
template = Model(
    n_classes = 43,
    name = "basic-conv-net.tf",
    graph = tf.Graph(),
    seed = seed,
    inputs = dict(
        features = features_test,
        labels = labels_test
    )
)

# model
model = template()

# restore
print("restoring model")
model.initialize(restore=True)

# test
print("testing")
test_score = model.score()
print("test score: {}".format(test_score))
