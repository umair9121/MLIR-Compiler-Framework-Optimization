"""ToyAI: a tiny array language compiled through a custom MLIR dialect."""

from .emitter import emit_program
from .lexer import ToyAISyntaxError, tokenize
from .parser import parse
from .shapes import ToyAITypeError, check_program

__all__ = [
    "ToyAISyntaxError",
    "ToyAITypeError",
    "check_program",
    "emit_program",
    "parse",
    "tokenize",
]
