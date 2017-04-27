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

# obtener todas las imagenes (lento)
data_generator = dataset.training_set.random_batch_arrays_generator(32)
data_generator = cz.map(Dict(features = P[0], labels = P[1]), data_generator)

# create model template
template = Model(
    n_classes = 43,
    name = "basic-conv-net.tf",
    graph = tf.Graph(),
    seed = seed,
    inputs = dict(
        features = dict(shape = (None, 32, 32, 3)),
        labels = dict(shape = (None,), dtype = tf.uint8)
    )
)

# create model
model = template()

# initialize variables
model.initialize()

# fit
print("training")
model.fit(data_generator=data_generator, epochs=1000)

# save
print("saving model")
model.save()
