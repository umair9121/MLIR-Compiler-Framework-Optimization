import numpy as np
import pytest

from toyai import ToyAISyntaxError, ToyAITypeError, ast, emit_program, parse, tokenize
from toyai.lexer import TokenKind
from toyai.reference import conv2d_nchw, evaluate


def test_lexer_tokens_and_locations():
    toks = tokenize("def f(A: f32[2, 3]) -> f32[2,3] {\n  return relu(A); # hi\n}")
    kinds = [t.kind for t in toks]
    assert kinds[:4] == [TokenKind.KW_DEF, TokenKind.IDENT, TokenKind.LPAREN, TokenKind.IDENT]
    assert TokenKind.ARROW in kinds and kinds[-1] == TokenKind.EOF
    ret = next(t for t in toks if t.kind == TokenKind.KW_RETURN)
    assert (ret.loc.line, ret.loc.col) == (2, 3)


def test_lexer_numbers():
    toks = tokenize("1 2.5 3e2 -4.0e-1")
    assert [(t.kind, t.text) for t in toks[:-1]] == [
        (TokenKind.INT, "1"),
        (TokenKind.FLOAT, "2.5"),
        (TokenKind.FLOAT, "3e2"),
        (TokenKind.MINUS, "-"),
        (TokenKind.FLOAT, "4.0e-1"),
    ]


def test_lexer_rejects_bad_char():
    with pytest.raises(ToyAISyntaxError, match=r"1:5: error: unexpected character '\$'"):
        tokenize("def $")


def test_operator_precedence():
    prog = parse("def f(A: f32[2,2], B: f32[2,2], C: f32[2,2]) -> f32[2,2] { return A @ B + C; }")
    ret = prog.functions[0].body[0].value
    assert isinstance(ret, ast.BinOp) and ret.op == "+"
    assert isinstance(ret.lhs, ast.BinOp) and ret.lhs.op == "@"


def test_parse_kwargs_and_lists():
    prog = parse(
        "def f(X: f32[1,1,4,4], K: f32[1,1,3,3]) -> f32[1,1,2,2] { return conv2d(X, K, stride=[2, 2], padding=1); }"
    )
    call = prog.functions[0].body[0].value
    assert call.callee == "conv2d" and set(call.kwargs) == {"stride", "padding"}


@pytest.mark.parametrize(
    "src, message",
    [
        ("def f(A: f32[2]) -> f32[2] { return A }", "expected ';'"),
        ("def f(A: i8[2]) -> f32[2] { return A; }", "unsupported element type 'i8'"),
        ("def f(A: f32[0]) -> f32[2] { return A; }", "dimension sizes must be positive"),
        ("def f(A: f32[2]) -> f32[2] { return relu(A, k=1, A); }", "positional argument follows keyword"),
    ],
)
def test_syntax_errors(src, message):
    with pytest.raises(ToyAISyntaxError, match=message):
        parse(src)


@pytest.mark.parametrize(
    "src, message",
    [
        ("def f(A: f32[2,3], B: f32[2,3]) -> f32[2,3] { return A @ B; }", "contraction mismatch"),
        ("def f(A: f32[2,3]) -> f32[2,3] { return relu(Z); }", "undefined value 'Z'"),
        ("def f(A: f32[2,3]) -> f32[3,2] { return relu(A); }", "declares f32\\[3, 2\\]"),
        ("def f(A: f32[2,3]) -> f32[2,3] { B = A; B = A; return B; }", "already defined"),
        ("def f(A: f32[2,3]) -> f32[2,3] { return softmax(A); }", "unknown operation 'softmax'"),
        ("def f(A: f32[2,3]) -> f32[2,3] { B = relu(A); }", "no 'return'"),
        ("def f(X: f32[1,3,8,8], K: f32[4,2,3,3]) -> f32[1,4,6,6] { return conv2d(X, K); }", "channel mismatch"),
        ("def f(A: f32[2,3], B: f32[3,2]) -> f32[2,3] { return A + B; }", "same type"),
    ],
)
def test_type_errors(src, message):
    with pytest.raises(ToyAITypeError, match=message):
        emit_program(parse(src))


def test_type_error_reports_location():
    src = "def f(A: f32[2,3]) -> f32[2,3] {\n  B = relu(A);\n  return relu(C);\n}"
    with pytest.raises(ToyAITypeError) as info:
        emit_program(parse(src, "model.tai"))
    assert str(info.value).startswith("model.tai:3:15:")


def test_emit_mlp():
    src = "def mlp(X: f32[4,8], W: f32[8,2], B: f32[4,2]) -> f32[4,2] { H = X @ W + B; return relu(H); }"
    mlir = emit_program(parse(src))
    assert "func.func @mlp(%X: tensor<4x8xf32>, %W: tensor<8x2xf32>, %B: tensor<4x2xf32>)" in mlir
    assert "attributes {llvm.emit_c_interface}" in mlir
    assert "%0 = toyai.matmul %X, %W : (tensor<4x8xf32>, tensor<8x2xf32>) -> tensor<4x2xf32>" in mlir
    assert "%1 = toyai.add %0, %B" in mlir
    assert "%2 = toyai.relu %1 : (tensor<4x2xf32>) -> tensor<4x2xf32>" in mlir
    assert "return %2 : tensor<4x2xf32>" in mlir


def test_emit_conv_and_constant():
    src = (
        "def f(X: f32[1,2,8,8], K: f32[3,2,3,3]) -> f32[1,3,4,4] {"
        "  Y = conv2d(X, K, stride=2, padding=1); Z = full([1,3,4,4], 0.5); return Y + Z; }"
    )
    mlir = emit_program(parse(src))
    assert "toyai.conv2d %X, %K {padding = array<i64: 1, 1>, strides = array<i64: 2, 2>}" in mlir
    assert "toyai.constant 5.000000000e-01 : tensor<1x3x4x4xf32>" in mlir


def test_conv_reference_matches_loops():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 3, 7, 6)).astype(np.float32)
    w = rng.standard_normal((4, 3, 3, 2)).astype(np.float32)
    got = conv2d_nchw(x, w, strides=(2, 1), padding=(1, 0))
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (0, 0)))
    oh, ow = (7 + 2 - 3) // 2 + 1, (6 - 2) // 1 + 1
    want = np.zeros((2, 4, oh, ow), np.float32)
    for n in range(2):
        for f in range(4):
            for i in range(oh):
                for j in range(ow):
                    want[n, f, i, j] = np.sum(xp[n, :, 2 * i: 2 * i + 3, j: j + 2] * w[f])
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_reference_evaluator():
    src = "def f(A: f32[3,4], B: f32[4,2]) -> f32[3,2] { C = A @ B; return relu(C + full([3,2], -1.0)); }"
    prog = parse(src)
    emit_program(prog)
    a = np.arange(12, dtype=np.float32).reshape(3, 4) / 10
    b = np.ones((4, 2), np.float32)
    np.testing.assert_allclose(evaluate(prog.functions[0], [a, b]), np.maximum(a @ b - 1, 0))
