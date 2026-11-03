"""End-to-end: .tai -> MLIR -> LLVM -> DLL, executed and compared to NumPy."""

import numpy as np
import pytest

from toyai import parse
from toyai.compiler import Compiler, CompileError, CpuOptions, Toolchain
from toyai.reference import evaluate

try:
    Toolchain().tool("toyai-opt")
    HAVE_TOOLCHAIN = True
except CompileError:
    HAVE_TOOLCHAIN = False

pytestmark = pytest.mark.skipif(not HAVE_TOOLCHAIN, reason="toyai-opt not built")

PROGRAMS = {
    "matmul_square": "def f(A: f32[128,128], B: f32[128,128]) -> f32[128,128] { return A @ B; }",
    "matmul_odd": "def f(A: f32[30,50], B: f32[50,70]) -> f32[30,70] { return matmul(A, B); }",
    "matmul_tall": "def f(A: f32[512,64], B: f32[64,256]) -> f32[512,256] { return A @ B; }",
    "mlp": (
        "def f(X: f32[64,256], W: f32[256,128], B: f32[64,128]) -> f32[64,128] {"
        "  H = X @ W + B; return relu(H); }"
    ),
    "chain": (
        "def f(A: f32[32,64], B: f32[64,32], C: f32[32,16]) -> f32[32,16] {"
        "  T = relu(A @ B); return relu(relu(T @ C)); }"
    ),
    "constant": "def f(A: f32[16,24]) -> f32[16,24] { return relu(A + full([16,24], -0.25)); }",
    "conv_same": (
        "def f(X: f32[1,16,32,32], K: f32[32,16,3,3]) -> f32[1,32,32,32] {"
        "  return relu(conv2d(X, K, padding=1)); }"
    ),
    "conv_strided": (
        "def f(X: f32[2,3,17,15], K: f32[4,3,5,3]) -> f32[2,4,9,8] {"
        "  return conv2d(X, K, stride=2, padding=[2, 1]); }"
    ),
    "vector_1d": "def f(A: f32[1000], B: f32[1000]) -> f32[1000] { return relu(A + B); }",
}


@pytest.fixture(scope="module")
def compiler():
    return Compiler()


@pytest.mark.parametrize("optimize", [False, True], ids=["naive", "optimized"])
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_cpu_matches_numpy(compiler, name, optimize):
    program = parse(PROGRAMS[name])
    module = compiler.compile(program, target="cpu", cpu=CpuOptions(optimize=optimize))
    fn = program.functions[0]
    rng = np.random.default_rng(42)
    inputs = [rng.standard_normal(p.type.shape).astype(np.float32) for p in fn.params]
    got = module["f"](*inputs)
    want = evaluate(fn, inputs)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_out_parameter_is_reused(compiler):
    program = parse("def f(A: f32[8,8], B: f32[8,8]) -> f32[8,8] { return A @ B; }")
    module = compiler.compile(program)
    a = np.eye(8, dtype=np.float32)
    b = np.arange(64, dtype=np.float32).reshape(8, 8)
    out = np.full((8, 8), np.nan, dtype=np.float32)
    result = module["f"](a, b, out=out)
    assert result is out
    np.testing.assert_array_equal(out, b)


def test_shape_check_at_call(compiler):
    program = parse("def f(A: f32[4,4]) -> f32[4,4] { return relu(A); }")
    module = compiler.compile(program)
    with pytest.raises(ValueError, match="must have shape"):
        module["f"](np.zeros((3, 4), np.float32))
