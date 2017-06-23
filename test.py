from __future__ import print_function

from dataget import data # <== dataget
import tensorflow as tf
import cytoolz as cz
from phi.api import *
from model import Model
import numpy as np
import random
from name import network_name, model_path
from tfinterface.supervised import SupervisedInputs
import click
from utils import batch_predict
from sklearn.metrics import accuracy_score

@click.command()
@click.option('--device', '-d', default="/gpu:0", help='Device, default = gpu:0')
def main(device):
    print("DEVICE:", device)

    # seed: resultados repetibles
    seed = 34
    np.random.seed(seed=seed)
    random.seed(seed)

    # dataget
    dataset = data("german-traffic-signs").get()

    # obtener imagenes
    print("loading data")
    features_test, labels_test = dataset.test_set.arrays()
    # features_test, labels_test = next(dataset.test_set.random_batch_arrays_generator(500))


    #model
    with tf.device(device):
        graph = tf.Graph()
        sess = tf.Session(graph=graph)

        # inputs
        inputs = SupervisedInputs(
            name = network_name + "_inputs",
            graph = graph,
            sess = sess,
            # tensors
            features = dict(shape = (None, 32, 32, 3)),
            labels = dict(shape = (None,), dtype = tf.uint8)
        )

        # create model template
        template = Model(
            n_classes = 43,
            name = network_name,
            model_path = model_path,
            graph = graph,
            sess = sess,
            seed = seed,
        )


        inputs = inputs()
        model = template(inputs)

        # restore
        print("restoring model")
        model.initialize(restore=True)

        # test
        print("testing")
        predictions = batch_predict(
            model, features_test, 100,
            print_fn = lambda batch: print(model.score(features=features_test[batch], labels=labels_test[batch]))
        )
        test_score = accuracy_score(predictions, labels_test)
        print("test score: {}".format(test_score))

if __name__ == '__main__':
    main()
