# https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook

import numpy as np
import struct
from array import array

def read_images(filepath :str):
    with open(filepath, "rb") as f:
        magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            print("WRONG MAGIC NUMBER, SOMETHING IS WRONG WITH MNIST DATASET FILE")
            return None
        image_data = array("B", f.read())

    images = []
    for i in range(size):
        img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
        images.append(img / 255)
    return images
