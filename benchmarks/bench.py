"""Benchmarks the ToyAI pipelines against NumPy.

    python benchmarks/bench.py [--quick] [--out docs]

Variants:
  cpu-naive    toyai-cpu-pipeline{optimize=false}: straight affine loops
  cpu-opt      cache tiling + vectorization + register blocking (1 thread)
  gpu          toyai-gpu-pipeline on the CUDA device, end to end
               (host->device copies, kernels, device->host copy)
  numpy-1t     NumPy/OpenBLAS restricted to one thread
  numpy        NumPy/OpenBLAS with all threads

Results are written to <out>/benchmarks.json and <out>/benchmarks.png.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from toyai import parse  # noqa: E402
from toyai.compiler import Compiler, CpuOptions  # noqa: E402
from toyai.reference import conv2d_nchw  # noqa: E402


def matmul_case(n: int) -> dict:
    return {
        "name": f"matmul {n}",
        "src": f"def k(A: f32[{n},{n}], B: f32[{n},{n}]) -> f32[{n},{n}] {{ return A @ B; }}",
        "shapes": [(n, n), (n, n)],
        "flops": 2 * n**3,
        "numpy": "a @ b",
        "naive": n <= 1024,
    }


def conv_case(n, c, h, f, k) -> dict:
    return {
        "name": f"conv {c}->{f} {h}x{h} k{k}",
        "src": (
            f"def k(X: f32[{n},{c},{h},{h}], W: f32[{f},{c},{k},{k}]) -> f32[{n},{f},{h},{h}] {{"
            f"  return conv2d(X, W, padding={k // 2}); }}"
        ),
        "shapes": [(n, c, h, h), (f, c, k, k)],
        "flops": 2 * n * f * h * h * c * k * k,
        "numpy": f"conv2d_nchw(a, b, padding=({k // 2}, {k // 2}))",
        "naive": True,
    }


def time_call(fn, reps: int, warmup_s: float = 0.5) -> float:
    """Best of `reps` runs after a warm-up (module load, page faults, CUDA
    context). The warm-up lasts at least `warmup_s` because an idle laptop GPU
    sits at a few hundred MHz and needs a moment of load to clock up."""
    t_end = time.perf_counter() + warmup_s
    fn()
    while time.perf_counter() < t_end:
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.min(samples))


NUMPY_SNIPPET = """
import sys, time, json
import numpy as np
sys.path.insert(0, {pydir!r})
from toyai.reference import conv2d_nchw
rng = np.random.default_rng(0)
a, b = (rng.standard_normal(s).astype(np.float32) for s in {shapes!r})
f = lambda: {expr}
f()
ts = []
for _ in range({reps}):
    t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
print(json.dumps(float(np.min(ts))))
"""


def time_numpy(case: dict, reps: int, threads: int | None) -> float:
    env = dict(os.environ)
    if threads is not None:
        env.update(OPENBLAS_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    code = NUMPY_SNIPPET.format(pydir=str(REPO / "python"), shapes=case["shapes"], expr=case["numpy"], reps=reps)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def run(cases: list[dict], reps: int) -> list[dict]:
    compiler = Compiler()
    rng = np.random.default_rng(0)
    results = []
    for case in cases:
        program = parse(case["src"])
        a, b = (rng.standard_normal(s).astype(np.float32) for s in case["shapes"])
        want = a @ b if case["name"].startswith("matmul") else conv2d_nchw(
            a, b, padding=(case["shapes"][1][2] // 2,) * 2)
        row = {"case": case["name"], "gflop": case["flops"] / 1e9, "seconds": {}}

        variants = [("cpu-opt", "cpu", CpuOptions(optimize=True)), ("gpu", "gpu", None)]
        if case["naive"]:
            variants.insert(0, ("cpu-naive", "cpu", CpuOptions(optimize=False)))
        for label, target, cpu in variants:
            kwargs = {"cpu": cpu} if cpu else {}
            module = compiler.compile(program, target=target, **kwargs)
            out = np.empty(want.shape, np.float32)
            module["k"](a, b, out=out)
            err = float(np.max(np.abs(out - want)) / max(1.0, float(np.max(np.abs(want)))))
            assert err < 1e-4, f"{label} {case['name']}: relative error {err}"
            row["seconds"][label] = time_call(lambda: module["k"](a, b, out=out),
                                              reps if label != "cpu-naive" else 1)
        row["seconds"]["numpy-1t"] = time_numpy(case, reps, threads=1)
        row["seconds"]["numpy"] = time_numpy(case, reps, threads=None)
        row["gflops"] = {k: case["flops"] / v / 1e9 for k, v in row["seconds"].items()}
        results.append(row)
        print(f"{case['name']:<24}" + "  ".join(f"{k}={v:8.1f}" for k, v in row["gflops"].items()) + "  GFLOP/s")
    return results


def plot(results: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["cpu-naive", "cpu-opt", "numpy-1t", "gpu", "numpy"]
    colors = {"cpu-naive": "#bbbbbb", "cpu-opt": "#4c72b0", "numpy-1t": "#8fb0d8",
              "gpu": "#55a868", "numpy": "#c44e52"}
    names = {"cpu-naive": "ToyAI CPU naive", "cpu-opt": "ToyAI CPU tiled+vectorized (1 thread)",
             "numpy-1t": "NumPy/OpenBLAS (1 thread)", "gpu": "ToyAI GPU (incl. transfers)",
             "numpy": "NumPy/OpenBLAS (20 threads)"}
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.16
    x = np.arange(len(results))
    for i, label in enumerate(labels):
        vals = [r["gflops"].get(label, 0.0) for r in results]
        bars = ax.bar(x + (i - 2) * width, vals, width, label=names[label], color=colors[label])
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.annotate(f"{v:.0f}" if v >= 10 else f"{v:.1f}", (bar.get_x() + bar.get_width() / 2, v),
                            ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, [r["case"] for r in results])
    ax.set_ylabel("GFLOP/s (higher is better)")
    ax.set_yscale("log")
    ax.set_title("ToyAI MLIR pipeline - i7-12700H / RTX 3050 Ti Laptop")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller problem set")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", type=Path, default=REPO / "docs")
    args = ap.parse_args()

    sizes = [256, 512] if args.quick else [256, 512, 1024, 2048]
    cases = [matmul_case(n) for n in sizes]
    cases.append(conv_case(1, 64, 56, 64, 3))
    if not args.quick:
        cases.append(conv_case(1, 128, 28, 128, 3))

    results = run(cases, args.reps)
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"cpu": platform.processor(), "python": platform.python_version(), "numpy": np.__version__}
    (args.out / "benchmarks.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    plot(results, args.out / "benchmarks.png")
    print(f"wrote {args.out / 'benchmarks.json'} and {args.out / 'benchmarks.png'}")


if __name__ == "__main__":
    main()
