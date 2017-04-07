import os, sys
from PIL import Image

dims = sys.argv[1] if len(sys.argv) > 1 else "32x32"
dims = dims.split('x')
dims = tuple(map(int, dims))

print("Image dims: {}".format(dims))

CLASS = None

for root, dirs, files in os.walk("data"):
    for file in files:
        file = os.path.join(root, file)

        if file.endswith(".ppm"):

            jpg_file = file.replace(".ppm", ".jpg")

            with Image.open(file) as im:
                im = im.resize(dims)
                im.save(jpg_file, quality=100)

            os.remove(file)

            dirs = file.split("/")

            _class = dirs[-2]
            _set = dirs[-6]

            if _class != CLASS:
                CLASS = _class

                print("formating {_set} class {_class}".format(_set = _set, _class = _class))

        elif file.endswith(".csv"):

            with open(file, 'r') as f:
                csv = f.read()

            csv = csv.replace(".ppm", ".jpg")

            with open(file, 'w') as f:
                f.write(csv)
