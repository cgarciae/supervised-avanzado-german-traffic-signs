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

@click.command()
@click.option('--device', '-d', default="/gpu:0", help='Device, default = gpu:0')
@click.option('--log', is_flag=True, help='Log network structure to tensorboard')
def main(device, log):

    graph = tf.Graph()
    sess = tf.Session(graph=graph)

    # inputs
    inputs = SupervisedInputs(
        name = network_name + "_inputs",
        graph = graph,
        sess = sess,
        model_path = model_path + ".count",
        # tensors
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
    with tf.device(device):
        inputs = inputs()
        model = template(inputs)

        if log:
            print("Writing Logs")
            writer = tf.summary.FileWriter(logdir='logs', graph=graph)
            writer.flush()

    with graph.as_default():
        print("")
        print("##########################################################")
        print("Number of Weights = {:,}".format(model.count_weights()))
        print("##########################################################")

        print(model.predictions)

if __name__ == '__main__':
    main()
