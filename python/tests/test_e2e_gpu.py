"""End-to-end on the GPU: .tai -> gpu/nvvm kernels + host code -> DLL."""

import ctypes

import numpy as np
import pytest

from toyai import parse
from toyai.compiler import Compiler, GpuOptions
from toyai.reference import evaluate

from test_e2e_cpu import HAVE_TOOLCHAIN, PROGRAMS


def _have_cuda() -> bool:
    try:
        cuda = ctypes.WinDLL("nvcuda.dll")
    except OSError:
        return False
    count = ctypes.c_int(0)
    return cuda.cuInit(0) == 0 and cuda.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value > 0


pytestmark = pytest.mark.skipif(
    not (HAVE_TOOLCHAIN and _have_cuda()), reason="needs toyai-opt and an NVIDIA GPU"
)


@pytest.fixture(scope="module")
def compiler():
    return Compiler()


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_gpu_matches_numpy(compiler, name):
    program = parse(PROGRAMS[name])
    module = compiler.compile(program, target="gpu")
    fn = program.functions[0]
    rng = np.random.default_rng(7)
    inputs = [rng.standard_normal(p.type.shape).astype(np.float32) for p in fn.params]
    got = module["f"](*inputs)
    np.testing.assert_allclose(got, evaluate(fn, inputs), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shared", [True, False], ids=["shared-memory", "simple"])
@pytest.mark.parametrize("m,k,n", [(64, 16, 64), (128, 96, 256), (256, 512, 192)])
def test_matmul_kernels(compiler, shared, m, k, n):
    program = parse(f"def f(A: f32[{m},{k}], B: f32[{k},{n}]) -> f32[{m},{n}] {{ return A @ B; }}")
    module = compiler.compile(program, target="gpu", gpu=GpuOptions(shared_matmul=shared))
    rng = np.random.default_rng(11)
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)
    np.testing.assert_allclose(module["f"](a, b), a @ b, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("block", [(8, 8), (32, 8), (16, 16)])
def test_block_shapes(compiler, block):
    program = parse("def f(A: f32[100,60], B: f32[60,130]) -> f32[100,130] { return relu(A @ B); }")
    module = compiler.compile(program, target="gpu", gpu=GpuOptions(block_x=block[0], block_y=block[1]))
    rng = np.random.default_rng(3)
    a = rng.standard_normal((100, 60)).astype(np.float32)
    b = rng.standard_normal((60, 130)).astype(np.float32)
    np.testing.assert_allclose(module["f"](a, b), np.maximum(a @ b, 0), rtol=1e-4, atol=1e-4)


def test_repeated_calls_reuse_module(compiler):
    program = parse("def f(A: f32[64,64], B: f32[64,64]) -> f32[64,64] { return A + B; }")
    module = compiler.compile(program, target="gpu")
    for i in range(20):
        a = np.full((64, 64), i, np.float32)
        np.testing.assert_array_equal(module["f"](a, a), 2 * a)
