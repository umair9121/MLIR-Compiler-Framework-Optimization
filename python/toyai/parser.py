"""Recursive-descent parser for the ToyAI array language.

Grammar (EBNF):

    program   ::= function* EOF
    function  ::= 'def' IDENT '(' [param (',' param)*] ')' '->' type '{' stmt* '}'
    param     ::= IDENT ':' type
    type      ::= IDENT '[' INT (',' INT)* ']'
    stmt      ::= IDENT '=' expr ';'
                | 'return' expr ';'
    expr      ::= term ('+' term)*            # elementwise add
    term      ::= primary ('@' primary)*      # matrix multiply
    primary   ::= number
                | '[' [expr (',' expr)*] ']'
                | IDENT '(' [arg (',' arg)*] ')'
                | IDENT
                | '(' expr ')'
    arg       ::= IDENT '=' expr | expr
    number    ::= ['-'] (INT | FLOAT)
"""

from __future__ import annotations

from . import ast
from .lexer import Token, TokenKind, ToyAISyntaxError, tokenize

SUPPORTED_DTYPES = {"f32"}


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    @property
    def tok(self) -> Token:
        return self.tokens[self.pos]

    def _peek(self, offset: int = 1) -> Token:
        i = min(self.pos + offset, len(self.tokens) - 1)
        return self.tokens[i]

    def _check(self, kind: TokenKind) -> bool:
        return self.tok.kind == kind

    def _accept(self, kind: TokenKind) -> Token | None:
        if self._check(kind):
            tok = self.tok
            self.pos += 1
            return tok
        return None

    def _expect(self, kind: TokenKind, what: str) -> Token:
        tok = self._accept(kind)
        if tok is None:
            found = self.tok.text or "end of file"
            raise ToyAISyntaxError(f"expected {what}, found {found!r}", self.tok.loc)
        return tok

    def parse_program(self) -> ast.Program:
        functions = []
        while not self._check(TokenKind.EOF):
            functions.append(self.parse_function())
        if not functions:
            raise ToyAISyntaxError("program contains no functions", self.tok.loc)
        return ast.Program(functions)

    def parse_function(self) -> ast.Function:
        start = self._expect(TokenKind.KW_DEF, "'def'")
        name = self._expect(TokenKind.IDENT, "function name").text
        self._expect(TokenKind.LPAREN, "'('")
        params: list[ast.Param] = []
        if not self._check(TokenKind.RPAREN):
            params.append(self.parse_param())
            while self._accept(TokenKind.COMMA):
                params.append(self.parse_param())
        self._expect(TokenKind.RPAREN, "')'")
        self._expect(TokenKind.ARROW, "'->'")
        ret = self.parse_type()
        self._expect(TokenKind.LBRACE, "'{'")
        body: list[ast.Stmt] = []
        while not self._check(TokenKind.RBRACE):
            if self._check(TokenKind.EOF):
                raise ToyAISyntaxError("unterminated function body, expected '}'", self.tok.loc)
            body.append(self.parse_stmt())
        self._expect(TokenKind.RBRACE, "'}'")
        return ast.Function(start.loc, name, params, ret, body)

    def parse_param(self) -> ast.Param:
        tok = self._expect(TokenKind.IDENT, "parameter name")
        self._expect(TokenKind.COLON, "':'")
        return ast.Param(tok.loc, tok.text, self.parse_type())

    def parse_type(self) -> ast.TensorType:
        dtype_tok = self._expect(TokenKind.IDENT, "element type")
        if dtype_tok.text not in SUPPORTED_DTYPES:
            raise ToyAISyntaxError(
                f"unsupported element type {dtype_tok.text!r} (supported: {', '.join(sorted(SUPPORTED_DTYPES))})",
                dtype_tok.loc,
            )
        self._expect(TokenKind.LBRACKET, "'['")
        dims = [self._parse_dim()]
        while self._accept(TokenKind.COMMA):
            dims.append(self._parse_dim())
        self._expect(TokenKind.RBRACKET, "']'")
        return ast.TensorType(dtype_tok.text, tuple(dims))

    def _parse_dim(self) -> int:
        tok = self._expect(TokenKind.INT, "dimension size")
        value = int(tok.text)
        if value <= 0:
            raise ToyAISyntaxError("dimension sizes must be positive", tok.loc)
        return value

    def parse_stmt(self) -> ast.Stmt:
        if ret := self._accept(TokenKind.KW_RETURN):
            value = self.parse_expr()
            self._expect(TokenKind.SEMI, "';'")
            return ast.Return(ret.loc, value)
        target = self._expect(TokenKind.IDENT, "statement")
        self._expect(TokenKind.ASSIGN, "'='")
        value = self.parse_expr()
        self._expect(TokenKind.SEMI, "';'")
        return ast.Assign(target.loc, target.text, value)

    def parse_expr(self) -> ast.Expr:
        lhs = self.parse_term()
        while op := self._accept(TokenKind.PLUS):
            lhs = ast.BinOp(op.loc, "+", lhs, self.parse_term())
        return lhs

    def parse_term(self) -> ast.Expr:
        lhs = self.parse_primary()
        while op := self._accept(TokenKind.AT):
            lhs = ast.BinOp(op.loc, "@", lhs, self.parse_primary())
        return lhs

    def parse_primary(self) -> ast.Expr:
        tok = self.tok
        if tok.kind in (TokenKind.INT, TokenKind.FLOAT, TokenKind.MINUS):
            return self._parse_number()
        if self._accept(TokenKind.LBRACKET):
            elements: list[ast.Expr] = []
            if not self._check(TokenKind.RBRACKET):
                elements.append(self.parse_expr())
                while self._accept(TokenKind.COMMA):
                    elements.append(self.parse_expr())
            self._expect(TokenKind.RBRACKET, "']'")
            return ast.ListLit(tok.loc, elements)
        if self._accept(TokenKind.LPAREN):
            inner = self.parse_expr()
            self._expect(TokenKind.RPAREN, "')'")
            return inner
        if self._accept(TokenKind.IDENT):
            if self._accept(TokenKind.LPAREN):
                return self._parse_call_rest(tok)
            return ast.Name(tok.loc, tok.text)
        raise ToyAISyntaxError(f"expected expression, found {tok.text or 'end of file'!r}", tok.loc)

    def _parse_number(self) -> ast.Number:
        start = self.tok
        negative = self._accept(TokenKind.MINUS) is not None
        tok = self.tok
        if tok.kind not in (TokenKind.INT, TokenKind.FLOAT):
            raise ToyAISyntaxError("expected a number after '-'", tok.loc)
        self.pos += 1
        value = float(tok.text)
        return ast.Number(start.loc, -value if negative else value, tok.kind == TokenKind.INT)

    def _parse_call_rest(self, callee: Token) -> ast.Call:
        args: list[ast.Expr] = []
        kwargs: dict[str, ast.Expr] = {}
        if not self._check(TokenKind.RPAREN):
            while True:
                if self._check(TokenKind.IDENT) and self._peek().kind == TokenKind.ASSIGN:
                    key = self.tok
                    self.pos += 2
                    if key.text in kwargs:
                        raise ToyAISyntaxError(f"duplicate keyword argument {key.text!r}", key.loc)
                    kwargs[key.text] = self.parse_expr()
                else:
                    if kwargs:
                        raise ToyAISyntaxError("positional argument follows keyword argument", self.tok.loc)
                    args.append(self.parse_expr())
                if not self._accept(TokenKind.COMMA):
                    break
        self._expect(TokenKind.RPAREN, "')'")
        return ast.Call(callee.loc, callee.text, args, kwargs)


def parse(source: str, filename: str = "<input>") -> ast.Program:
    return Parser(tokenize(source, filename)).parse_program()
