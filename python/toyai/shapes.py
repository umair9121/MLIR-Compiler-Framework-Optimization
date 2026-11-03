"""Name resolution and shape inference for ToyAI programs.

Every expression that produces a tensor gets its `type` field set. Errors carry
the source location of the offending expression.
"""

from __future__ import annotations

from . import ast
from .lexer import Location


class ToyAITypeError(Exception):
    def __init__(self, message: str, loc: Location):
        super().__init__(f"{loc}: error: {message}")
        self.loc = loc
        self.message = message


def _int_list(expr: ast.Expr, what: str, length: int) -> tuple[int, ...]:
    """Accepts `n` or `[a, b, ...]` and returns a tuple of `length` ints."""
    if isinstance(expr, ast.Number):
        values = [expr]
    elif isinstance(expr, ast.ListLit):
        values = expr.elements
    else:
        raise ToyAITypeError(f"{what} must be an integer or a list of integers", expr.loc)
    out = []
    for v in values:
        if not isinstance(v, ast.Number) or not v.is_int:
            raise ToyAITypeError(f"{what} must contain integers", v.loc)
        out.append(int(v.value))
    if len(out) == 1:
        out = out * length
    if len(out) != length:
        raise ToyAITypeError(f"{what} must have {length} entries, got {len(out)}", expr.loc)
    return tuple(out)


def conv2d_output_shape(
    input_shape: tuple[int, ...],
    filter_shape: tuple[int, ...],
    strides: tuple[int, int],
    padding: tuple[int, int],
) -> tuple[int, ...]:
    n, _, h, w = input_shape
    f, _, kh, kw = filter_shape
    oh = (h + 2 * padding[0] - kh) // strides[0] + 1
    ow = (w + 2 * padding[1] - kw) // strides[1] + 1
    return (n, f, oh, ow)


class ShapeChecker:
    def __init__(self, fn: ast.Function):
        self.fn = fn
        self.env: dict[str, ast.TensorType] = {}

    def check(self) -> None:
        for p in self.fn.params:
            if p.name in self.env:
                raise ToyAITypeError(f"duplicate parameter {p.name!r}", p.loc)
            self.env[p.name] = p.type

        returned = False
        for stmt in self.fn.body:
            if returned:
                raise ToyAITypeError("statement after 'return' is unreachable", stmt.loc)
            ty = self.expr(stmt.value)
            if isinstance(stmt, ast.Assign):
                if stmt.target in self.env:
                    raise ToyAITypeError(f"{stmt.target!r} is already defined (values are immutable)", stmt.loc)
                self.env[stmt.target] = ty
            else:
                if ty != self.fn.return_type:
                    raise ToyAITypeError(
                        f"returned value has type {ty}, but function {self.fn.name!r} declares {self.fn.return_type}",
                        stmt.value.loc,
                    )
                returned = True
        if not returned:
            raise ToyAITypeError(f"function {self.fn.name!r} has no 'return' statement", self.fn.loc)

    def tensor(self, expr: ast.Expr) -> ast.TensorType:
        ty = self.expr(expr)
        if ty is None:
            raise ToyAITypeError("expected a tensor value", expr.loc)
        return ty

    def expr(self, expr: ast.Expr) -> ast.TensorType | None:
        if isinstance(expr, ast.Name):
            if expr.name not in self.env:
                raise ToyAITypeError(f"use of undefined value {expr.name!r}", expr.loc)
            expr.type = self.env[expr.name]
        elif isinstance(expr, ast.BinOp):
            callee = "add" if expr.op == "+" else "matmul"
            expr.type = self._builtin(callee, [expr.lhs, expr.rhs], {}, expr.loc)
        elif isinstance(expr, ast.Call):
            expr.type = self._builtin(expr.callee, expr.args, expr.kwargs, expr.loc)
        elif isinstance(expr, (ast.Number, ast.ListLit)):
            raise ToyAITypeError("scalar and list literals are only allowed as builtin arguments", expr.loc)
        return expr.type

    def _arity(self, name: str, args: list, kwargs: dict, n: int, allowed_kw: set[str], loc: Location) -> None:
        if len(args) != n:
            raise ToyAITypeError(f"{name}() takes {n} positional argument(s), got {len(args)}", loc)
        for key, value in kwargs.items():
            if key not in allowed_kw:
                raise ToyAITypeError(f"{name}() got an unexpected keyword argument {key!r}", value.loc)

    def _builtin(self, name: str, args: list[ast.Expr], kwargs: dict[str, ast.Expr], loc: Location) -> ast.TensorType:
        if name == "matmul":
            self._arity(name, args, kwargs, 2, set(), loc)
            a, b = self.tensor(args[0]), self.tensor(args[1])
            if len(a.shape) != 2 or len(b.shape) != 2:
                raise ToyAITypeError(f"matmul expects rank-2 operands, got {a} and {b}", loc)
            if a.shape[1] != b.shape[0]:
                raise ToyAITypeError(
                    f"matmul contraction mismatch: {a} @ {b} ({a.shape[1]} != {b.shape[0]})", loc
                )
            return ast.TensorType(a.dtype, (a.shape[0], b.shape[1]))

        if name == "add":
            self._arity(name, args, kwargs, 2, set(), loc)
            a, b = self.tensor(args[0]), self.tensor(args[1])
            if a != b:
                raise ToyAITypeError(f"add expects operands of the same type, got {a} and {b}", loc)
            return a

        if name == "relu":
            self._arity(name, args, kwargs, 1, set(), loc)
            return self.tensor(args[0])

        if name == "conv2d":
            self._arity(name, args, kwargs, 2, {"stride", "padding"}, loc)
            x, w = self.tensor(args[0]), self.tensor(args[1])
            if len(x.shape) != 4 or len(w.shape) != 4:
                raise ToyAITypeError(f"conv2d expects NCHW input and FCHW filter, got {x} and {w}", loc)
            if x.shape[1] != w.shape[1]:
                raise ToyAITypeError(
                    f"conv2d channel mismatch: input has {x.shape[1]} channels, filter expects {w.shape[1]}", loc
                )
            strides = _int_list(kwargs["stride"], "stride", 2) if "stride" in kwargs else (1, 1)
            padding = _int_list(kwargs["padding"], "padding", 2) if "padding" in kwargs else (0, 0)
            if min(strides) < 1:
                raise ToyAITypeError("stride must be >= 1", kwargs["stride"].loc)
            if min(padding) < 0:
                raise ToyAITypeError("padding must be >= 0", kwargs["padding"].loc)
            out = conv2d_output_shape(x.shape, w.shape, strides, padding)
            if out[2] < 1 or out[3] < 1:
                raise ToyAITypeError(f"conv2d filter {w} does not fit input {x} with padding {padding}", loc)
            return ast.TensorType(x.dtype, out)

        if name == "full":
            self._arity(name, args, kwargs, 2, set(), loc)
            shape_expr, value = args
            if not isinstance(shape_expr, ast.ListLit) or not shape_expr.elements:
                raise ToyAITypeError("full() expects a shape list like [4, 4] as first argument", shape_expr.loc)
            shape = _int_list(shape_expr, "shape", len(shape_expr.elements))
            if min(shape) < 1:
                raise ToyAITypeError("dimension sizes must be positive", shape_expr.loc)
            if not isinstance(value, ast.Number):
                raise ToyAITypeError("full() expects a numeric fill value", value.loc)
            return ast.TensorType("f32", shape)

        raise ToyAITypeError(
            f"unknown operation {name!r} (available: matmul, conv2d, add, relu, full)", loc
        )


def check_program(program: ast.Program) -> None:
    seen: set[str] = set()
    for fn in program.functions:
        if fn.name in seen:
            raise ToyAITypeError(f"redefinition of function {fn.name!r}", fn.loc)
        seen.add(fn.name)
        ShapeChecker(fn).check()
