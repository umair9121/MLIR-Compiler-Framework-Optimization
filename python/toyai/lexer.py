"""Tokenizer for the ToyAI array language (.tai files)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class TokenKind(Enum):
    IDENT = auto()
    INT = auto()
    FLOAT = auto()
    KW_DEF = auto()
    KW_RETURN = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    COLON = auto()
    SEMI = auto()
    ASSIGN = auto()
    ARROW = auto()
    PLUS = auto()
    MINUS = auto()
    AT = auto()
    EOF = auto()


KEYWORDS = {
    "def": TokenKind.KW_DEF,
    "return": TokenKind.KW_RETURN,
}

SINGLE_CHAR = {
    "(": TokenKind.LPAREN,
    ")": TokenKind.RPAREN,
    "[": TokenKind.LBRACKET,
    "]": TokenKind.RBRACKET,
    "{": TokenKind.LBRACE,
    "}": TokenKind.RBRACE,
    ",": TokenKind.COMMA,
    ":": TokenKind.COLON,
    ";": TokenKind.SEMI,
    "=": TokenKind.ASSIGN,
    "+": TokenKind.PLUS,
    "@": TokenKind.AT,
}


@dataclass(frozen=True)
class Location:
    line: int
    col: int
    filename: str = "<input>"

    def __str__(self) -> str:
        return f"{self.filename}:{self.line}:{self.col}"


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    text: str
    loc: Location


class ToyAISyntaxError(Exception):
    def __init__(self, message: str, loc: Location):
        super().__init__(f"{loc}: error: {message}")
        self.loc = loc
        self.message = message


class Lexer:
    def __init__(self, source: str, filename: str = "<input>"):
        self.src = source
        self.filename = filename
        self.pos = 0
        self.line = 1
        self.col = 1

    def _loc(self) -> Location:
        return Location(self.line, self.col, self.filename)

    def _peek(self, offset: int = 0) -> str:
        i = self.pos + offset
        return self.src[i] if i < len(self.src) else ""

    def _advance(self) -> str:
        ch = self.src[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def _skip_trivia(self) -> None:
        while self.pos < len(self.src):
            ch = self._peek()
            if ch in " \t\r\n":
                self._advance()
            elif ch == "#":
                while self.pos < len(self.src) and self._peek() != "\n":
                    self._advance()
            else:
                break

    def _number(self, loc: Location) -> Token:
        start = self.pos
        is_float = False
        while self._peek().isdigit():
            self._advance()
        if self._peek() == "." and self._peek(1).isdigit():
            is_float = True
            self._advance()
            while self._peek().isdigit():
                self._advance()
        if self._peek() in ("e", "E"):
            nxt = self._peek(1)
            if nxt.isdigit() or (nxt in "+-" and self._peek(2).isdigit()):
                is_float = True
                self._advance()
                if self._peek() in "+-":
                    self._advance()
                while self._peek().isdigit():
                    self._advance()
        text = self.src[start:self.pos]
        return Token(TokenKind.FLOAT if is_float else TokenKind.INT, text, loc)

    def tokens(self) -> list[Token]:
        out: list[Token] = []
        while True:
            self._skip_trivia()
            loc = self._loc()
            if self.pos >= len(self.src):
                out.append(Token(TokenKind.EOF, "", loc))
                return out
            ch = self._peek()
            if ch.isalpha() or ch == "_":
                start = self.pos
                while self._peek().isalnum() or self._peek() == "_":
                    self._advance()
                text = self.src[start:self.pos]
                out.append(Token(KEYWORDS.get(text, TokenKind.IDENT), text, loc))
            elif ch.isdigit():
                out.append(self._number(loc))
            elif ch == "-" and self._peek(1) == ">":
                self._advance()
                self._advance()
                out.append(Token(TokenKind.ARROW, "->", loc))
            elif ch == "-":
                self._advance()
                out.append(Token(TokenKind.MINUS, "-", loc))
            elif ch in SINGLE_CHAR:
                self._advance()
                out.append(Token(SINGLE_CHAR[ch], ch, loc))
            else:
                raise ToyAISyntaxError(f"unexpected character {ch!r}", loc)


def tokenize(source: str, filename: str = "<input>") -> list[Token]:
    return Lexer(source, filename).tokens()
