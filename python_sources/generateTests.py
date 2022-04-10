from testsGenerator import *
import numpy as np
import random
import subprocess

DELTA = 0.0001

# 	matrix create tests

gen = TestsGenerator()
for pair in [[0, 0], [1, 0], [0, 1]]:
    gen.genTest(
        "mx_create",
        "\tmx_t *a = mx_create({}, {});\n".format(*pair) + genAssert("a != NULL"),
    )

x, y = random.randint(1, 16), random.randint(1, 16)
gen.genTest(
    "mx_create",
    f"\tmx_t *a = mx_create({x}, {y});\n"
    + genAssert("a == NULL")
    + genAssert(f"a->x != {x} || a->y != {y} || a->size != {x * y}")
    + "\tmx_destroy(a);",
)

# 	matrix multiplication tests

for t in ("0", "A", "B", "BOTH"):
    for n in range(5):
        x1, y1 = random.randint(1, 16), random.randint(1, 16)
        x2, y2 = {
            "0": (random.randint(1, 16), x1),
            "A": (random.randint(1, 16), y1),
            "B": (x1, random.randint(1, 16)),
            "BOTH": (y1, random.randint(1, 16)),
        }[t]

        x3, y3 = {"0": (x2, y1), "A": (x2, x1), "B": (y2, y1), "BOTH": (y2, x1)}[t]
        a = np.random.randint(-254, high=254, size=(y1, x1))
        b = np.random.randint(-254, high=254, size=(y2, x2))
        expected = np.matmul(
            np.transpose(a) if t in ("A", "BOTH") else a,
            np.transpose(b) if t in ("B", "BOTH") else b,
        ).flatten()

        gen.genTest(
            "mx_mp" if t == "0" else "mx_mp (transposed)",
            genStaticMxDec(a, "a")
            + genStaticMxDec(b, "b")
            + genStaticEmptyMxDec((y3, x3), "res")
            + genStaticListDec(expected, "exp")
            + f"\tmx_mp(a, b, &res, {t});\n\n"
            + genMxComp("res", "exp", DELTA),
        )

# 	matrix hadamard product tests

for n in range(5):
    x, y = random.randint(1, 16), random.randint(1, 16)
    a = np.random.randint(-254, high=256, size=(y, x))
    b = np.random.randint(-254, high=256, size=(y, x))
    expected = np.multiply(a, b).flatten()
    gen.genTest(
        "mx_hadamard",
        genStaticMxDec(a, "a")
        + genStaticMxDec(b, "b")
        + genStaticEmptyMxDec((y, x), "res")
        + genStaticListDec(expected, "exp")
        + f"\tmx_hadamard(a, b, &res);\n"
        + genMxComp("res", "exp", DELTA),
    )

# 	matrix substraction

for n in range(5):
    x, y = random.randint(1, 16), random.randint(1, 16)
    a = np.random.randint(-254, high=254, size=(y, x))
    b = np.random.randint(-254, high=254, size=(y, x))
    expected = (a - b).flatten()
    gen.genTest(
        "mx_substraction",
        genStaticMxDec(a, "a")
        + genStaticMxDec(b, "b")
        + genStaticEmptyMxDec((y, x), "res")
        + genStaticListDec(expected, "exp")
        + f"\tmx_sub(a, b, &res);\n"
        + genMxComp("res", "exp", DELTA),
    )

# 	matrix multiply by number

for n in range(5):
    x, y = random.randint(1, 16), random.randint(1, 16)
    a = np.random.randint(-254, high=254, size=(y, x))
    num = random.randint(-16, 16)
    expected = (a * num).flatten()
    gen.genTest(
        "mx_mp_num",
        genStaticMxDec(a, "a")
        + genStaticListDec(expected, "exp")
        + f"\tmx_mp_num(&a, {num});\n"
        + genMxComp("a", "exp", DELTA),
    )

# 	matrix hadamard product with lambda

gen.appendCode("\nmx_type foo(mx_type a) {return (a > 4) ? 0 : a;}\n")

x, y = random.randint(1, 16), random.randint(1, 16)
a = np.random.randint(-254, high=254, size=(y, x))
b = np.random.randint(-254, high=254, size=(y, x))
new_b = np.array([[0 if x > 4 else x for x in y] for y in b])
expected = (a * new_b).flatten()

gen.genTest(
    "mx_hadam_lambda",
    genStaticMxDec(a, "a")
    + genStaticMxDec(b, "b")
    + f"\tmx_hadam_lambda(&a, b, (&foo));\n"
    + genStaticListDec(expected, "exp")
    + genMxComp("a", "exp", DELTA),
)

# 	neural network create

for n in range(5):
    inputSize = random.randint(1, 16)
    batchSize = random.randint(1, 16)
    denseSize1 = random.randint(1, 16)
    denseSize2 = random.randint(1, 16)
    maxTmp = max([inputSize * denseSize1, denseSize1 * denseSize2])
    gen.genTest(
        "nn_create",
        # initializer of neural network
        f"""
		nn_t *nn = nn_create({inputSize}, {batchSize}, 0.01);
		LAYER_DENSE(nn, {denseSize1}, RELU, 0.1, 0.2);
		LAYER_DENSE(nn, {denseSize2}, NO_FUNC, 0.1, 0.2);\n"""
        +
        # check size of first layer
        genAssert(
            f"nn->layers[0].out->x != {denseSize1} || nn->layers[0].out->y != {batchSize}"
        )
        + genAssert(f"nn->layers[0].weights->x != {inputSize} || nn->layers[0].weights->y != {denseSize1}")
        +
        # check size of second layer
        genAssert("nn->layers[1].out == NULL")
        + genAssert(
            f"nn->layers[1].out->x != {denseSize2} || nn->layers[1].out->y != {batchSize}"
        )
        + genAssert(f"nn->layers[1].weights->x != {denseSize1} || nn->layers[1].weights->y != {denseSize2}")
        +
        # check size of temporary matrix stored in neural network struct
        genAssert(f"nn->temp->size != {maxTmp}") + "\tnn_destroy(nn);\n",
    )

# 	neural network predict

gen.genTest(
    "nn_predict",
    genStaticMxDec(np.array([[8.5, 0.65, 1.2]]), "input")
    + """
	nn_t *n = nn_create(3, 1, 0.01);
	LAYER_DENSE(n, 3, NO_FUNC, 0.0, 0.0);
	LAYER_DENSE(n, 3, NO_FUNC, 0.0, 0.0);\n"""
    + genStaticListDec([0.1, 0.2, -0.1, -0.1, 0.1, 0.9, 0.1, 0.4, 0.1], "val0")
    + genStaticListDec([0.3, 1.1, -0.3, 0.1, 0.2, 0.0, 0.0, 1.3, 0.1], "val1")
    + genListCpy("val0", "n->layers[0].weights->arr", "n->layers[0].weights->size")
    + genListCpy("val1", "n->layers[1].weights->arr", "n->layers[1].weights->size")
    + "\tnn_predict(n, &input);\n"
    + genStaticListDec([0.2135, 0.145, 0.5065], "expected")
    + genMxComp("(*n->layers[1].out)", "expected", DELTA)
    + "\tnn_destroy(n);\n",
)

gen.genTest(
    "nn_predict",
    """	nn_t *nn = nn_create(1, 1, 0.01);
	LAYER_DENSE(nn, 3, RELU, 0.0, 0.0);
	LAYER_DENSE(nn, 1, NO_FUNC, 0.0, 0.0);\n"""
    + genStaticListDec([0.1, -0.1, 0.1], "val0")
    + genStaticListDec([0.3, 1.1, -0.3], "val1")
    + genListCpy("val0", "nn->layers[0].weights->arr", "nn->layers[0].weights->size")
    + genListCpy("val1", "nn->layers[1].weights->arr", "nn->layers[1].weights->size")
    + genStaticMxDec(np.array([[8.5]]), "input")
    + "\tnn_predict(nn, &input);\n"
    + genAssert(
        f"nn->layers[1].out->arr[0] > {DELTA} || nn->layers[1].out->arr[0] < -{DELTA}"
    )
    + "\tnn_destroy(nn);\n",
)

##	neural network predict

gen.genTest(
    "nn_fit",
    """   	nn_t* nn = nn_create(3, 4, 0.01);
	LAYER_DENSE(nn, 3, RELU, 0.0, 0.0);
	LAYER_DENSE(nn, 3, NO_FUNC, 0.0, 0.0);\n"""
    + genStaticMxDec(
        np.array([[8.5, 0.65, 1.2], [9.5, 0.8, 1.3], [9.9, 0.8, 0.5], [9.0, 0.9, 1.0]]),
        "input",
    )
    + genStaticListDec([0.1, 0.2, -0.1, -0.1, 0.1, 0.9, 0.1, 0.4, 0.1], "val0")
    + genStaticListDec([0.3, 1.1, -0.3, 0.1, 0.2, 0.0, 0.0, 1.3, 0.1], "val1")
    + genListCpy("val0", "nn->layers[0].weights->arr", "nn->layers[0].weights->size")
    + genListCpy("val1", "nn->layers[1].weights->arr", "nn->layers[1].weights->size")
    + genStaticMxDec(
        np.array([[0.1, 1.0, 0.1], [0.0, 1.0, 0.0], [0.0, 0.0, 0.1], [0.1, 1.0, 0.2]]),
        "out",
    )
    + "\tnn_fit(nn, &input, &out);\n"
    + genStaticListDec(
        [
            0.118847,
            0.201724,
            -0.0977926,
            -0.190674,
            0.0930147,
            0.886871,
            0.093963,
            0.399449,
            0.0994944,
        ],
        "exp_val0",
    )
    + genStaticListDec(
        [
            0.29901,
            1.09916,
            -0.301627,
            0.123058,
            0.205844,
            0.0328309,
            -0.0096053,
            1.29716,
            0.0863697,
        ],
        "exp_val1",
    )
    + genMxComp("(*nn->layers[1].weights)", "exp_val1", DELTA)
    + genMxComp("(*nn->layers[0].weights)", "exp_val0", DELTA)
    + "\tnn_destroy(nn);\n",
)

"""gen.genTest("im2col",
    genStaticMxDec(np.array([
        [1, 3, 1], 
        [2, 6, 6],
        [3, 0, 4],
        [4, 8, 5],
        [8, 2, 3],
        [7, 4, 0],
        [6, 3, 9],
        [5, 3, 8],
        [3, 4, 1],
        [7, 6, 0],
        [2, 2, 3],
        [1, 5, 1]]), "input")
    + genStaticEmptyMxDec((12, 4), "output")
    + "im2col(&input, &output, 3, 2, 3, 2, 2, 2);"
    + genStaticListDec(
        [1, 2, 4, 8, 3, 6, 8, 2, 1, 6, 5, 3,
        2, 3, 8, 7, 6, 0, 2, 4, 6, 4, 3, 0,
        6, 5, 7, 2, 3, 3, 6, 2, 9, 8, 0, 3,
        5, 3, 2, 1, 3, 4, 2, 5, 8, 1, 3, 1], "expected")
    + genMxComp("output", "expected", DELTA))
"""
gen.save("sources/main.c")
