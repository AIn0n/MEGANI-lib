from read_mnist import read_images
from testsGenerator import *
import numpy as np
import random

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
		nn_t *nn = nn_create({inputSize}, {batchSize});
        add_batch_gradient_descent(nn, 0.01);
		LAYER_DENSE(nn, {denseSize1}, RELU, 0.1, 0.2);
		LAYER_DENSE(nn, {denseSize2}, NO_FUNC, 0.1, 0.2);\n"""
        +
        # check size of first layer
        genAssert(
            f"nn->layers[0].out->x != {denseSize1} || nn->layers[0].out->y != {batchSize}"
        )
        + genAssert(
            f"nn->layers[0].weights->x != {inputSize} || nn->layers[0].weights->y != {denseSize1}"
        )
        +
        # check size of second layer
        genAssert("nn->layers[1].out == NULL")
        + genAssert(
            f"nn->layers[1].out->x != {denseSize2} || nn->layers[1].out->y != {batchSize}"
        )
        + genAssert(
            f"nn->layers[1].weights->x != {denseSize1} || nn->layers[1].weights->y != {denseSize2}"
        )
        +
        # check size of temporary matrix stored in neural network struct
        genAssert(f"nn->temp->size != {maxTmp}") + "\tnn_destroy(nn);\n",
    )

# 	neural network predict

gen.genTest(
    "nn_predict",
    genStaticMxDec(np.array([[8.5, 0.65, 1.2]]), "input")
    + """
	nn_t *n = nn_create(3, 1);
    add_batch_gradient_descent(n, 0.01);
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
    """
    nn_t *nn = nn_create(1, 1);
    add_batch_gradient_descent(nn, 0.01);
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
    """
    nn_t* nn = nn_create(3, 4);
    add_batch_gradient_descent(nn, 0.01);
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

size = random.randint(1, 16)
arr = [np.array([[random.randint(0, 16)]]) for _ in range(size)]
indexes = [str(n) for n in range(len(arr))]

gen.genTest(
    "default matrix list iterator test",
    "".join(genStaticMxDec(arr[int(n)], "a" + n) for n in indexes)
    + "mx_t* list[] = {"
    + "".join([f"&a{n}, " for n in indexes])
    + "};"
    + """
    mx_iterator iterator = {.list = list, .size = """
    + str(size)
    + """, .curr = 0};
    void * iter = (void *) &iterator;
    """
    + "".join(genAssert("default_iter_next(iter) != &a" + n) for n in indexes)
)

size = random.randint(1, 256)
gen.genTest("default matrix list iterator has_next function test",
    "".join(genStaticMxDec(arr[int(n)], "a" + n) for n in indexes)
    + "mx_t* list[] = {"
    + "".join([f"&a{n}, " for n in indexes])
    + "};"
    + f"""
    mx_iterator iterator = {{.list = list, .size = {size}, .curr = 0}};
    void *iter = (void *) &iterator;
    int i = 0;
    do {{
        default_iter_next(iter);
    }} while (default_iter_has_next(iter));
    """
    + genAssert(f"i == {size}")
)

mnist_images_filepath = "mnist/t10k-images-idx3-ubyte"
mnist = read_images(mnist_images_filepath)

# TODO: check more indexes
index = random.randint(0, 9_999)
gen.genTest("reading idx3 file test - random image from mnist",
    genStaticListDec(mnist[index], "expected")
+f"""
        mx_iterator iterator = read_idx3("{mnist_images_filepath}", 1, 1);
        void *iter = (void *) &iterator;
        mx_t result = *(iterator.list[{index}]);
"""
+ genMxComp("result", "expected", DELTA)
+ """
        do {
                mx_destroy(default_iter_next(iter));
        } while(default_iter_has_next(iter));
        free(iterator.list);
""")

inputSize = random.randint(1, 16)
batchSize = random.randint(1, 16)
denseSize1 = random.randint(1, 16)
denseSize2 = random.randint(1, 16)
maxTmp = max([inputSize * denseSize1, denseSize1 * denseSize2])
gen.genTest(
    "rms prop creation - how much and how large caches do we get",
    # initializer of neural network
    f"""
    nn_t *nn = nn_create({inputSize}, {batchSize});
    LAYER_DENSE(nn, {denseSize1}, RELU, 0.1, 0.2);
    LAYER_DENSE(nn, {denseSize2}, NO_FUNC, 0.1, 0.2);
    add_rms_prop(nn, 0.01, 0.9);
    rms_prop_data_t *data = (void *) nn->optimizer.params;\n"""
    # check size of first rms cache
    + genAssert(
        f"data->caches[0]->x != {inputSize} || data->caches[0]->y != {denseSize1}"
    )
    # check size of second rms cache
    + genAssert(
        f"data->caches[1]->x != {denseSize1} || data->caches[1]->y != {denseSize2}"
    )
)

x = random.randint(1, 16)
y = random.randint(1, 16)
mx = np.random.randint(-254, high=254, size=(y, x))
expected = mx ** 2

gen.genTest("test matrix element-wise power by factor of 2",
    genStaticMxDec(mx, "mx")
    + genStaticListDec(expected.flatten(), "expected") + """
        mx_elem_power_by_two(&mx);\n"""
    + genMxComp("mx", "expected", DELTA)
)

x = random.randint(1, 16)
y = random.randint(1, 16)
in_out_mx = np.random.randint(-254, high=254, size=(y, x))
mx_to_add = np.random.randint(-254, high=254, size=(y, x))
expected = in_out_mx + mx_to_add

gen.genTest("mx_add_to_first test",
    genStaticMxDec(in_out_mx, "in_out_mx")
    + genStaticMxDec(mx_to_add, "mx_to_add")
    + genStaticListDec(expected.flatten(), "expected") + """
        mx_add_to_first(&in_out_mx, &mx_to_add);\n"""
    + genMxComp("in_out_mx", "expected", DELTA)
)

mx = np.array([[4, 9], [16, 25]])
expected = np.sqrt(mx).flatten()

gen.genTest("element-wise sqrt",
    genStaticMxDec(mx, "mx")
    + genStaticListDec(expected, "expected") + """
        mx_cell_sqrt(&mx);\n"""
    + genMxComp("mx", "expected", DELTA))

gen.save("sources/main.c")
