
def batch_generator(total, batch_size):

    i = 0

    while i < total:

        _from = i
        _to = min(i + batch_size, total)

        yield list(range(_from, _to))

        i += batch_size


def batch_predict(model, features, labels, batch_size):

    preds_list = []

    for batch in batch_generator(len(features), batch_size):
        preds = model.predict(features=features[batch])
        preds = np.argmax(preds, axis=1)
        preds_list.append(preds)

    return np.concatenate(preds_list, axis=0)

if __name__ == '__main__':
    print(list(batch_generator(10, 3)))
