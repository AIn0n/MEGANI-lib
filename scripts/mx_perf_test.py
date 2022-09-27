import random
import numpy as np
from generate_functional_tests import DELTA
from testsGenerator import *

gen = TestsGenerator()

for t in ("0", "A", "B", "BOTH"):
    for ranges in [(16, 32), (32, 48), (48, 64), (64, 128), (128, 255), (255, 512)]:
        x1, y1 = random.randint(*ranges), random.randint(*ranges)
        x2, y2 = {
            "0": (random.randint(*ranges), x1),
            "A": (random.randint(*ranges), y1),
            "B": (x1, random.randint(*ranges)),
            "BOTH": (y1, random.randint(*ranges)),
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

gen.save("sources/perf_tests.c")