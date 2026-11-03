"""AST node definitions for the ToyAI array language."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from .lexer import Location


@dataclass(frozen=True)
class TensorType:
    dtype: str
    shape: tuple[int, ...]

    def mlir(self) -> str:
        dims = "x".join(str(d) for d in self.shape)
        return f"tensor<{dims}x{self.dtype}>" if dims else f"tensor<{self.dtype}>"

    def __str__(self) -> str:
        return f"{self.dtype}[{', '.join(str(d) for d in self.shape)}]"


@dataclass
class Expr:
    loc: Location
    # Filled in by shape inference (shapes.py).
    type: Optional[TensorType] = field(default=None, init=False, compare=False)


@dataclass
class Number(Expr):
    value: float
    is_int: bool


@dataclass
class ListLit(Expr):
    elements: list[Expr]


@dataclass
class Name(Expr):
    name: str


@dataclass
class Call(Expr):
    callee: str
    args: list[Expr]
    kwargs: dict[str, Expr]


@dataclass
class BinOp(Expr):
    op: str  # "+" or "@"
    lhs: Expr
    rhs: Expr


@dataclass
class Assign:
    loc: Location
    target: str
    value: Expr


@dataclass
class Return:
    loc: Location
    value: Expr


Stmt = Union[Assign, Return]


@dataclass
class Param:
    loc: Location
    name: str
    type: TensorType


@dataclass
class Function:
    loc: Location
    name: str
    params: list[Param]
    return_type: TensorType
    body: list[Stmt]


@dataclass
class Program:
    functions: list[Function]
