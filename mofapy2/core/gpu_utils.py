import sys
import numpy as np

try:
    import cupy as cp
except ImportError:
    pass

# --------------------------------------------------------
# Defalut is to use numpy, not cupy
# --------------------------------------------------------
gpu_mode = False


# --------------------------------------------------------
# operations on single matrices
# --------------------------------------------------------
def log(mat):
    if gpu_mode:
        return cp.log(mat)
    else:
        return np.log(mat)


def square(mat):
    if gpu_mode:
        return cp.square(mat)
    else:
        return np.square(mat)


def exp(mat):
    if gpu_mode:
        return cp.exp(mat)
    else:
        return np.exp(mat)


def sum(mat):
    if gpu_mode:
        return cp.sum(mat)
    else:
        return np.sum(mat)


def sigmoid(mat):
    if gpu_mode:
        return cp.divide(1.0, 1.0 + cp.exp(-mat))
    else:
        return np.divide(1.0, 1.0 + np.exp(-mat))


def sqrt(mat):
    if gpu_mode:
        return cp.sqrt(mat)
    else:
        return np.sqrt(mat)


def tanh(mat):
    if gpu_mode:
        return cp.tanh(mat)
    else:
        return np.tanh(mat)


# --------------------------------------------------------
# operations on matrices pairs
# --------------------------------------------------------
def dot(mat1, mat2):
    if gpu_mode:
        return cp.dot(mat1, mat2)
    else:
        return np.dot(mat1, mat2)


def divide(mat1, mat2):
    if gpu_mode:
        return cp.divide(mat1, mat2)
    else:
        return np.divide(mat1, mat2)


# --------------------------------------------------------
# initialiser
# --------------------------------------------------------
def zeros(dim):
    if gpu_mode:
        return cp.zeros(dim)
    else:
        return np.zeros(dim)


# --------------------------------------------------------
# loading on and from the GPU
# --------------------------------------------------------
# optiMOFA: data matrices registered here are uploaded once per fit and the device copy is
# reused (stock re-uploads Y in every W, Z and Tau update). Keyed by id(); the entry is
# dropped when the host array is garbage collected, and cleared at the start of every fit.
_resident = {}


def register_resident(mat):
    import weakref

    key = id(mat)
    _resident[key] = (weakref.ref(mat), None)
    weakref.finalize(mat, _resident.pop, key, None)


def clear_resident():
    _resident.clear()


def array(mat):
    if gpu_mode:
        hit = _resident.get(id(mat))
        if hit is not None and hit[0]() is mat:
            if hit[1] is None:
                hit = (hit[0], cp.array(mat))
                _resident[id(mat)] = hit
            return hit[1]
        return cp.array(mat)
    else:
        return mat


def asnumpy(mat):
    if gpu_mode:
        return cp.asnumpy(mat)
    else:
        return mat
