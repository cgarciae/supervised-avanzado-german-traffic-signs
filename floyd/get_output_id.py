#! /usr/bin/python

import re
import sys


def main():
    text = " ".join(sys.argv)
    output_id_match = re.search(r"Output ID\s+(\w+)", text)

    if output_id_match:
        output = output_id_match.group(1)
        print(output)

    else:
        raise Exception("Unable to find output id in: {}".format(text))


if __name__ == '__main__':
    main()
