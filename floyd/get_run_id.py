#! /usr/bin/python

import sys
import re

def main():
    text = " ".join(sys.argv)
    run_id_match = re.search(r"floyd logs (\w+)", text)

    if run_id_match:
        run_id = run_id_match.group(1)
        print(run_id)

    else:
        raise Exception("Unable to find run id in: {}".format(text))


if __name__ == '__main__':
    main()
