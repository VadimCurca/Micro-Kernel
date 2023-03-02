import ctypes
import pathlib

import numpy as np
import torch.nn.functional as F
import torch

def printResult(test, passed):
    if (passed):
        print(test, ": [PASS]")
    else:
        print(test, ": [FAIL]")


def testLinear(c_lib):
    c_lib.linear_impl.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
    ]
    c_lib.linear_impl.restype = ctypes.c_void_p

    in_features = 64
    out_features = 128
    input = np.random.rand(in_features).astype(np.float32)
    weights = np.random.rand(out_features, in_features).astype(np.float32)
    biases = np.random.rand(out_features).astype(np.float32)
    microKernel_linear = np.empty(shape=(out_features), dtype=np.float32)

    torch_linear = F.linear(torch.from_numpy(input), torch.from_numpy(weights), torch.from_numpy(biases)).numpy()
    c_lib.linear_impl(in_features, out_features, input, weights, biases, microKernel_linear)

    passed = np.allclose(torch_linear, microKernel_linear)

    printResult('testLinear', passed)


if __name__ == "__main__":
    libname = pathlib.Path().absolute() / "build/lib/kernels/libkernels.so"
    c_lib = ctypes.cdll.LoadLibrary(libname)

    testLinear(c_lib)
