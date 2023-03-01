import ctypes
import pathlib
import numpy as np

def addScalars(c_lib):
    n = c_lib.add(1, 2)
    print('n =', n)

def passPointerToBuffer_1(c_lib):
    values = [1, 4, 5, 3, 2]
    n = len(values)
    array = (ctypes.c_int * n)(*values)

    c_lib.increment(array, n)
    print(array[:])


def passPointerToBuffer_2(c_lib):
    values = np.array([-3, 4, 5, 3, 3], dtype=np.int32)
    n = len(values)

    c_lib.increment.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_size_t
    ]
    c_lib.increment.restype = ctypes.c_void_p

    c_lib.increment(values, n)
    print(values)


def addBuffers(c_lib):
    values = np.array([-3, 4, 5, 3, 3], dtype=np.int32)
    n = len(values)
    output = np.empty(shape=(n), dtype=np.int32)

    c_lib.add_buffers.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_size_t
    ]
    c_lib.add_buffers.restype = ctypes.c_void_p

    c_lib.add_buffers(values, values, output, n)
    print(output)

def passStruct(c_lib):

    class Point(ctypes.Structure):
        _fields_ = [
            ('x', ctypes.c_int),
            ('y', ctypes.c_int)
        ]

        def __repr__(self) -> str:
            return '({0}, {1})'.format(self.x, self.y)

    p = Point(2, 3)

    c_lib.incrementStruct.argtypes = [
        ctypes.POINTER(Point)
    ]
    c_lib.incrementStruct.restype = ctypes.c_void_p

    c_lib.incrementStruct(p)
    c_lib.incrementStruct(p)

    print(p)


if __name__ == "__main__":
    libname = pathlib.Path().absolute() / "build/init/libinit.so"
    c_lib = ctypes.cdll.LoadLibrary(libname)

    addScalars(c_lib)

    passPointerToBuffer_1(c_lib)
    passPointerToBuffer_2(c_lib)

    addBuffers(c_lib)

    passStruct(c_lib)

