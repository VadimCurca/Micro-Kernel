import ctypes
import numpy as np

class MicroTensor(ctypes.Structure):
    _fields_ = [
        ("Address", ctypes.c_void_p),
        ("Shape", ctypes.POINTER(ctypes.c_int32)),
        ("Size", ctypes.c_int32)
    ]

def fromNumpy(npArray : np.array):
    address = npArray.ctypes.data_as(ctypes.c_void_p)
    shape = np.array(npArray.shape, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    size = len(npArray.shape)
    return MicroTensor(address, shape, size)

