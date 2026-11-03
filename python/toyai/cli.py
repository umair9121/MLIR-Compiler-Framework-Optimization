"""Command-line entry point.

    python -m toyai emit   model.tai                 # print ToyAI dialect MLIR
    python -m toyai lower  model.tai --target cpu    # print lowered (LLVM dialect) MLIR
    python -m toyai run    model.tai --target gpu    # compile, run on random inputs, check vs NumPy
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .emitter import emit_program
from .lexer import ToyAISyntaxError
from .parser import parse
from .shapes import ToyAITypeError


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="toyai")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("emit", "lower", "run"):
        p = sub.add_parser(name)
        p.add_argument("file", type=Path)
        if name != "emit":
            p.add_argument("--target", choices=["cpu", "gpu"], default="cpu")
            p.add_argument("--no-opt", action="store_true", help="disable CPU tiling/vectorization")
    args = ap.parse_args(argv)

    source = args.file.read_text()
    try:
        program = parse(source, str(args.file))
        toyai_mlir = emit_program(program)
    except (ToyAISyntaxError, ToyAITypeError) as e:
        print(e, file=sys.stderr)
        return 1

    if args.cmd == "emit":
        sys.stdout.write(toyai_mlir)
        return 0

    from .compiler import Compiler, CompileError, CpuOptions, GpuOptions

    compiler = Compiler()
    cpu = CpuOptions(optimize=not args.no_opt)
    try:
        if args.cmd == "lower":
            pipeline = cpu.pipeline() if args.target == "cpu" else GpuOptions().pipeline()
            sys.stdout.write(compiler.lower(toyai_mlir, pipeline))
            return 0
        module = compiler.compile(program, target=args.target, cpu=cpu)
    except CompileError as e:
        print(e, file=sys.stderr)
        return 1

    from .reference import evaluate

    rng = np.random.default_rng(0)
    for fn in program.functions:
        inputs = [rng.standard_normal(p.type.shape).astype(np.float32) for p in fn.params]
        t0 = time.perf_counter()
        got = module[fn.name](*inputs)
        dt = time.perf_counter() - t0
        want = evaluate(fn, inputs)
        err = float(np.max(np.abs(got - want))) if got.size else 0.0
        status = "OK" if np.allclose(got, want, rtol=1e-3, atol=1e-3) else "MISMATCH"
        print(f"{fn.name}: {status}  shape={got.shape}  max|err|={err:.2e}  time={dt * 1e3:.3f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
