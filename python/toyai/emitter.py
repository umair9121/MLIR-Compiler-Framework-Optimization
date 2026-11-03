"""Emits textual ToyAI-dialect MLIR from a shape-checked AST."""

from __future__ import annotations

from . import ast
from .shapes import _int_list, check_program


def _float_literal(value: float) -> str:
    # MLIR float literals require a '.', so always use scientific notation.
    return f"{value:.9e}"


def _i64_array(values: tuple[int, ...]) -> str:
    return "array<i64: " + ", ".join(str(v) for v in values) + ">"


class FunctionEmitter:
    def __init__(self, fn: ast.Function, indent: str = "    "):
        self.fn = fn
        self.indent = indent
        self.lines: list[str] = []
        self.values: dict[str, str] = {}
        self.counter = 0

    def _fresh(self) -> str:
        name = f"%{self.counter}"
        self.counter += 1
        return name

    def _op(self, opname: str, operands: list[ast.Expr], result: ast.TensorType, attrs: str = "") -> str:
        ssa = [self.expr(o) for o in operands]
        name = self._fresh()
        in_types = ", ".join(o.type.mlir() for o in operands)
        attr_str = f" {{{attrs}}}" if attrs else ""
        self.lines.append(
            f"{self.indent}{name} = toyai.{opname} {', '.join(ssa)}{attr_str} : ({in_types}) -> {result.mlir()}"
        )
        return name

    def expr(self, e: ast.Expr) -> str:
        if isinstance(e, ast.Name):
            return self.values[e.name]
        if isinstance(e, ast.BinOp):
            return self._op("add" if e.op == "+" else "matmul", [e.lhs, e.rhs], e.type)
        if isinstance(e, ast.Call):
            if e.callee in ("matmul", "add", "relu"):
                return self._op(e.callee, e.args, e.type)
            if e.callee == "conv2d":
                strides = _int_list(e.kwargs["stride"], "stride", 2) if "stride" in e.kwargs else (1, 1)
                padding = _int_list(e.kwargs["padding"], "padding", 2) if "padding" in e.kwargs else (0, 0)
                attrs = f"padding = {_i64_array(padding)}, strides = {_i64_array(strides)}"
                return self._op("conv2d", e.args, e.type, attrs)
            if e.callee == "full":
                name = self._fresh()
                value = _float_literal(e.args[1].value)
                self.lines.append(f"{self.indent}{name} = toyai.constant {value} : {e.type.mlir()}")
                return name
        raise AssertionError(f"unhandled expression {e!r}")

    def emit(self) -> str:
        params = []
        for p in self.fn.params:
            ssa = f"%{p.name}"
            self.values[p.name] = ssa
            params.append(f"{ssa}: {p.type.mlir()}")
        for stmt in self.fn.body:
            value = self.expr(stmt.value)
            if isinstance(stmt, ast.Assign):
                self.values[stmt.target] = value
            else:
                self.lines.append(f"{self.indent}return {value} : {self.fn.return_type.mlir()}")
        header = (
            f"  func.func @{self.fn.name}({', '.join(params)}) -> {self.fn.return_type.mlir()} "
            f"attributes {{llvm.emit_c_interface}} {{"
        )
        return "\n".join([header, *self.lines, "  }"])


def emit_program(program: ast.Program) -> str:
    check_program(program)
    funcs = [FunctionEmitter(fn).emit() for fn in program.functions]
    return "module {\n" + "\n".join(funcs) + "\n}\n"
