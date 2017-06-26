from PIL import Image
import numpy as np
from random import random

def random_rotate_feature(feature, rotation):
    angle = rotation * (random() * 2.0 - 1.0)
    im = Image.fromarray(feature).rotate(angle)

    return np.asarray(im)

def batch_random_image_rotation(generator, rotation):

    for features, labels in generator:

        features_list = map(lambda feature: random_rotate_feature(feature, rotation), features)
        features = np.stack(features_list, axis=0)

        yield features, labels



def batch_generator(total, batch_size):

    i = 0

    while i < total:

        _from = i
        _to = min(i + batch_size, total)

        yield list(range(_from, _to))

        i += batch_size


def batch_predict(model, features, batch_size, print_fn=None, **kwargs):

    preds_list = []

    for batch in batch_generator(len(features), batch_size):
        preds = model.predict(features=features[batch], **kwargs)
        preds = np.argmax(preds, axis=1)
        preds_list.append(preds)

        if print_fn:
            print_fn(batch)

    return np.concatenate(preds_list, axis=0)

if __name__ == '__main__':
    print(list(batch_generator(10, 3)))
