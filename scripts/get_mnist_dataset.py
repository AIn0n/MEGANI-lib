import os, requests, gzip

DATASET_FOLDER = "mnist"
yann_lecun_site = "http://yann.lecun.com/exdb/mnist/"
filenames = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)

if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER)

for filename in filenames:
    path = DATASET_FOLDER + "/" + filename
    print(path)
    open(path, "wb").write(requests.get(yann_lecun_site + filename).content)
    print("\tfile saved")
    with gzip.open(path, "rb") as compressed_f, open(path[:-3], "wb") as uncompressed_f:
        uncompressed_f.write(compressed_f.read())
    print("\tfile decompressed")
    os.remove(path)
