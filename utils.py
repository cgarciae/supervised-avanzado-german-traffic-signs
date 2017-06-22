
def batch_generator(total, batch_size):

    i = 0

    while i < total:

        _from = i
        _to = min(i + batch_size, total)

        yield list(range(_from, _to))

        i += batch_size


if __name__ == '__main__':
    print(list(batch_generator(10, 3)))
