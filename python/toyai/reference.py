"""NumPy reference interpreter for ToyAI functions (the correctness oracle)."""

from __future__ import annotations

import numpy as np

from . import ast
from .shapes import _int_list


def conv2d_nchw(x: np.ndarray, w: np.ndarray, strides=(1, 1), padding=(0, 0)) -> np.ndarray:
    x = np.pad(x, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])))
    n, c, h, wd = x.shape
    f, _, kh, kw = w.shape
    oh = (h - kh) // strides[0] + 1
    ow = (wd - kw) // strides[1] + 1
    windows = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(2, 3))
    windows = windows[:, :, :: strides[0], :: strides[1]][:, :, :oh, :ow]
    return np.einsum("nchwkl,fckl->nfhw", windows, w, optimize=True).astype(np.float32)


def evaluate(fn: ast.Function, inputs: list[np.ndarray]) -> np.ndarray:
    env = {p.name: np.asarray(a, dtype=np.float32) for p, a in zip(fn.params, inputs)}

    def ev(e: ast.Expr) -> np.ndarray:
        if isinstance(e, ast.Name):
            return env[e.name]
        if isinstance(e, ast.BinOp):
            lhs, rhs = ev(e.lhs), ev(e.rhs)
            return (lhs + rhs) if e.op == "+" else (lhs @ rhs)
        if isinstance(e, ast.Call):
            if e.callee == "matmul":
                return ev(e.args[0]) @ ev(e.args[1])
            if e.callee == "add":
                return ev(e.args[0]) + ev(e.args[1])
            if e.callee == "relu":
                return np.maximum(ev(e.args[0]), np.float32(0))
            if e.callee == "full":
                return np.full(e.type.shape, e.args[1].value, dtype=np.float32)
            if e.callee == "conv2d":
                strides = _int_list(e.kwargs["stride"], "stride", 2) if "stride" in e.kwargs else (1, 1)
                padding = _int_list(e.kwargs["padding"], "padding", 2) if "padding" in e.kwargs else (0, 0)
                return conv2d_nchw(ev(e.args[0]), ev(e.args[1]), strides, padding)
        raise AssertionError(f"unhandled expression {e!r}")

    for stmt in fn.body:
        value = ev(stmt.value)
        if isinstance(stmt, ast.Return):
            return value.astype(np.float32)
        env[stmt.target] = value
    raise AssertionError("function has no return")
