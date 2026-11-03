"""Loads compiled ToyAI DLLs and calls them with NumPy arrays.

Functions are compiled with `llvm.emit_c_interface` and results are turned
into out-parameters, so `def f(A, B) -> C` is exported as

    void _mlir_ciface_f(MemRef2D *A, MemRef2D *B, MemRef2D *C);

where each MemRef descriptor is the standard MLIR strided layout:

    struct { T *allocated; T *aligned; int64_t offset;
             int64_t sizes[rank]; int64_t strides[rank]; };
"""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import ast


@lru_cache(maxsize=None)
def memref_descriptor_type(rank: int) -> type[ctypes.Structure]:
    fields = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
    ]
    if rank > 0:
        fields += [("sizes", ctypes.c_int64 * rank), ("strides", ctypes.c_int64 * rank)]
    return type(f"MemRef{rank}D", (ctypes.Structure,), {"_fields_": fields})


def to_memref(array: np.ndarray) -> ctypes.Structure:
    if not array.flags["C_CONTIGUOUS"]:
        raise ValueError("arrays passed to ToyAI kernels must be C-contiguous")
    desc = memref_descriptor_type(array.ndim)()
    ptr = array.ctypes.data
    desc.allocated = ptr
    desc.aligned = ptr
    desc.offset = 0
    if array.ndim:
        desc.sizes[:] = array.shape
        desc.strides[:] = [s // array.itemsize for s in array.strides]
    return desc


_dll_dirs: list = []


def _register_dll_dirs(extra_dirs: list[Path]) -> None:
    for d in extra_dirs:
        if d.is_dir() and str(d) not in {str(h.path) for h in _dll_dirs if hasattr(h, "path")}:
            _dll_dirs.append(os.add_dll_directory(str(d)))


class CompiledFunction:
    def __init__(self, dll: ctypes.CDLL, fn: ast.Function):
        self.fn = fn
        self.name = fn.name
        self._cfunc = getattr(dll, f"_mlir_ciface_{fn.name}")
        self._cfunc.restype = None
        self._cfunc.argtypes = [ctypes.c_void_p] * (len(fn.params) + 1)

    def __call__(self, *arrays: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        if len(arrays) != len(self.fn.params):
            raise TypeError(f"{self.name}() takes {len(self.fn.params)} arguments, got {len(arrays)}")
        prepared = []
        for param, arr in zip(self.fn.params, arrays):
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            if arr.shape != param.type.shape:
                raise ValueError(
                    f"argument {param.name!r} of {self.name}() must have shape {param.type.shape}, got {arr.shape}"
                )
            prepared.append(arr)
        ret = self.fn.return_type
        if out is None:
            out = np.empty(ret.shape, dtype=np.float32)
        elif out.shape != ret.shape or out.dtype != np.float32 or not out.flags["C_CONTIGUOUS"]:
            raise ValueError(f"out must be a C-contiguous float32 array of shape {ret.shape}")

        descs = [to_memref(a) for a in prepared] + [to_memref(out)]
        self._cfunc(*[ctypes.addressof(d) for d in descs])
        return out


class CompiledModule:
    def __init__(self, dll_path: Path, program: ast.Program, dll_dirs: list[Path] = ()):
        _register_dll_dirs(list(dll_dirs))
        self.dll_path = Path(dll_path)
        self._dll = ctypes.CDLL(str(self.dll_path))
        self.functions = {fn.name: CompiledFunction(self._dll, fn) for fn in program.functions}

    def __getattr__(self, name: str) -> CompiledFunction:
        functions = self.__dict__.get("functions", {})
        if name in functions:
            return functions[name]
        raise AttributeError(name)

    def __getitem__(self, name: str) -> CompiledFunction:
        return self.functions[name]
