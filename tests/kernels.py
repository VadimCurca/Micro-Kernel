import ctypes
import pathlib

import numpy as np
import torch.nn.functional as F
import torch

from src.MicroTensor import *

def printResult(test, passed):
    if (passed):
        print(test, ": [PASS]")
    else:
        print(test, ": [FAIL]")


def testLinear(c_lib):
    c_lib.linear_impl.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(MicroTensor),
        ctypes.POINTER(MicroTensor),
        ctypes.POINTER(MicroTensor),
        ctypes.POINTER(MicroTensor),
    ]
    c_lib.linear_impl.restype = ctypes.c_void_p

    in_features = 64
    out_features = 128
    input = np.random.rand(in_features).astype(np.float32)
    weights = np.random.rand(out_features, in_features).astype(np.float32)
    biases = np.random.rand(out_features).astype(np.float32)
    microKernel_linear = np.empty(shape=(out_features), dtype=np.float32)

    torch_linear = F.linear(torch.from_numpy(input), torch.from_numpy(weights), torch.from_numpy(biases)).numpy()

    mt_input = fromNumpy(input)
    mt_weights = fromNumpy(weights)
    mt_biases = fromNumpy(biases)
    mt_output = fromNumpy(microKernel_linear)

    c_lib.linear_impl(in_features, out_features, mt_input, mt_weights, mt_biases, mt_output)

    passed = np.allclose(torch_linear, microKernel_linear)

    printResult('testLinear', passed)


if __name__ == "__main__":
    libname = pathlib.Path().absolute() / "build/lib/kernels/libkernels.so"
    c_lib = ctypes.cdll.LoadLibrary(libname)

    testLinear(c_lib)
