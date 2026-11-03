"""Compiler driver: .tai source -> ToyAI MLIR -> LLVM IR -> native DLL.

    source --parse/emit--> ToyAI MLIR
           --toyai-opt --toyai-{cpu,gpu}-pipeline--> LLVM dialect
           --mlir-translate--> LLVM IR
           --llc--> COFF object
           --link /DLL--> DLL loaded through ctypes (runtime.py)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from . import ast
from .emitter import emit_program
from .parser import parse
from .runtime import CompiledModule

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LLVM_INSTALL = Path(os.environ.get("TOYAI_LLVM_INSTALL", r"D:\U\toolchains\llvm-install"))
CACHE_DIR = Path(os.environ.get("TOYAI_CACHE_DIR", REPO_ROOT / ".toyai_cache"))


class CompileError(RuntimeError):
    pass


@dataclass(frozen=True)
class CpuOptions:
    """Options forwarded to `--toyai-cpu-pipeline`."""

    optimize: bool = True
    l1_kb: int = 48
    l2_kb: int = 1280
    unroll: int = 4
    vector_width: int = 8

    def pipeline(self) -> str:
        return (
            f"toyai-cpu-pipeline{{optimize={str(self.optimize).lower()} l1-kb={self.l1_kb} "
            f"l2-kb={self.l2_kb} unroll={self.unroll} vector-width={self.vector_width}}}"
        )


@dataclass(frozen=True)
class GpuOptions:
    """Options forwarded to `--toyai-gpu-pipeline`."""

    block_x: int = 16
    block_y: int = 16
    chip: str = "sm_86"

    def pipeline(self) -> str:
        return f"toyai-gpu-pipeline{{block-x={self.block_x} block-y={self.block_y} chip={self.chip}}}"


@dataclass
class Toolchain:
    llvm_install: Path = DEFAULT_LLVM_INSTALL
    build_dir: Path = REPO_ROOT / "build"
    extra_dll_dirs: list[Path] = field(default_factory=list)

    def tool(self, name: str) -> Path:
        candidates = [self.build_dir / "bin" / f"{name}.exe", self.llvm_install / "bin" / f"{name}.exe"]
        for c in candidates:
            if c.exists():
                return c
        found = shutil.which(name)
        if found:
            return Path(found)
        raise CompileError(f"could not find {name}; build the project or run scripts/env.ps1")

    @property
    def dll_dirs(self) -> list[Path]:
        dirs = [self.llvm_install / "bin", *self.extra_dll_dirs]
        cuda = os.environ.get("CUDA_PATH")
        if cuda:
            dirs.append(Path(cuda) / "bin")
        return dirs


@lru_cache(maxsize=None)
def msvc_environment() -> dict[str, str]:
    """Environment of an x64 MSVC developer prompt (needed for link.exe and the CRT libs)."""
    if os.environ.get("VSCMD_ARG_TGT_ARCH") == "x64":
        return dict(os.environ)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "msvc_env.json"
    if cache.exists():
        return json.loads(cache.read_text())
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / (
        r"Microsoft Visual Studio\Installer\vswhere.exe"
    )
    paths = subprocess.run(
        [str(vswhere), "-all", "-products", "*", "-property", "installationPath"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    vcvars = next(
        (Path(p.strip()) / r"VC\Auxiliary\Build\vcvars64.bat" for p in paths
         if (Path(p.strip()) / r"VC\Auxiliary\Build\vcvars64.bat").exists()),
        None,
    )
    if vcvars is None:
        raise CompileError("could not locate vcvars64.bat; install the MSVC C++ build tools")
    out = subprocess.run(
        f'cmd /c ""{vcvars}" >nul 2>&1 && set"', capture_output=True, text=True, shell=True, check=True
    ).stdout
    env = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    cache.write_text(json.dumps(env))
    return env


_STRUCTORS_RE = re.compile(r"^@llvm\.global_(ctors|dtors) = .*$", re.MULTILINE)


def explicit_initializers(llvm_ir: str) -> str:
    """Replaces llvm.global_ctors/dtors with exported toyai_init/toyai_fini.

    GPU modules are loaded by global constructors that call into the CUDA
    driver. In a DLL those run inside DllMain, under the Windows loader lock,
    where initializing CUDA deadlocks. The runtime calls toyai_init right after
    LoadLibrary instead.
    """
    funcs = {"ctors": [], "dtors": []}

    def strip(match: re.Match) -> str:
        funcs[match.group(1)] += re.findall(r"ptr @([\w.$]+)", match.group(0))
        return ""

    llvm_ir = _STRUCTORS_RE.sub(strip, llvm_ir)

    def wrapper(name: str, callees: list[str]) -> str:
        calls = "".join(f"  call void @{c}()\n" for c in callees)
        return f"\ndefine dllexport void @{name}() {{\n{calls}  ret void\n}}\n"

    return llvm_ir + wrapper("toyai_init", funcs["ctors"]) + wrapper("toyai_fini", funcs["dtors"][::-1])


def _run(cmd: list[str], what: str, env: dict | None = None, input_text: str | None = None) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, input=input_text)
    if proc.returncode != 0:
        raise CompileError(f"{what} failed:\n$ {' '.join(map(str, cmd))}\n{proc.stdout}{proc.stderr}")
    return proc.stdout


class Compiler:
    def __init__(self, toolchain: Toolchain | None = None, cache_dir: Path = CACHE_DIR):
        self.tc = toolchain or Toolchain()
        self.cache_dir = Path(cache_dir)

    def lower(self, mlir_text: str, pipeline: str) -> str:
        """Runs toyai-opt with the given pass pipeline and returns the resulting MLIR."""
        return _run(
            [str(self.tc.tool("toyai-opt")), f"--pass-pipeline=builtin.module({pipeline})", "-"],
            "toyai-opt", input_text=mlir_text,
        )

    def compile(
        self,
        source: str | ast.Program,
        target: str = "cpu",
        cpu: CpuOptions = CpuOptions(),
        gpu: GpuOptions = GpuOptions(),
        filename: str = "<input>",
    ) -> CompiledModule:
        program = parse(source, filename) if isinstance(source, str) else source
        toyai_mlir = emit_program(program)
        if target == "cpu":
            pipeline = cpu.pipeline()
        elif target == "gpu":
            pipeline = gpu.pipeline()
        else:
            raise ValueError(f"unknown target {target!r} (expected 'cpu' or 'gpu')")

        key = hashlib.sha256(f"{target}\n{pipeline}\n{toyai_mlir}".encode()).hexdigest()[:16]
        workdir = self.cache_dir / key
        dll = workdir / "module.dll"
        if not dll.exists():
            workdir.mkdir(parents=True, exist_ok=True)
            (workdir / "input.mlir").write_text(toyai_mlir)
            self._build(toyai_mlir, pipeline, program, target, workdir, dll)
        return CompiledModule(dll, program, self.tc.dll_dirs)

    def _build(self, toyai_mlir, pipeline, program, target, workdir: Path, dll: Path) -> None:
        llvm_dialect = self.lower(toyai_mlir, pipeline)
        (workdir / "lowered.mlir").write_text(llvm_dialect)

        ll = workdir / "module.ll"
        _run([str(self.tc.tool("mlir-translate")), "--mlir-to-llvmir", str(workdir / "lowered.mlir"),
              "-o", str(ll)], "mlir-translate")
        ll.write_text(explicit_initializers(ll.read_text()))

        obj = workdir / "module.obj"
        _run([str(self.tc.tool("llc")), "-O3", "-mcpu=alderlake", "--fp-contract=fast", "-filetype=obj",
              "-mtriple=x86_64-pc-windows-msvc", str(ll), "-o", str(obj)], "llc")

        libdir = self.tc.llvm_install / "lib"
        libs = ["msvcrt.lib", "ucrt.lib", "vcruntime.lib", "kernel32.lib",
                str(libdir / "mlir_c_runner_utils.lib")]
        if target == "gpu":
            libs.append(str(libdir / "mlir_cuda_runtime.lib"))
        exports = [f"/EXPORT:_mlir_ciface_{fn.name}" for fn in program.functions]
        env = msvc_environment()
        link = shutil.which("link.exe", path=env.get("PATH") or env.get("Path"))
        if link is None:
            raise CompileError("link.exe not found in the MSVC environment")
        _run([link, "/NOLOGO", "/DLL", f"/OUT:{dll}", str(obj), *exports, *libs], "link", env=env)


def compile_source(source: str, target: str = "cpu", **kwargs) -> CompiledModule:
    return Compiler().compile(source, target=target, **kwargs)


def compile_file(path: str | Path, target: str = "cpu", **kwargs) -> CompiledModule:
    path = Path(path)
    return Compiler().compile(path.read_text(), target=target, filename=str(path), **kwargs)
