# Builds and installs the pinned LLVM/MLIR toolchain used by ToyAI.
#
#   powershell -ExecutionPolicy Bypass -File scripts\build_llvm.ps1
#
# Produces <ToolchainRoot>\llvm-install with MLIR, the X86 and NVPTX backends,
# FileCheck & friends, and the mlir_cuda_runtime library used by GPU binaries.

param(
    [string]$ToolchainRoot = "D:\U\toolchains",
    [string]$LlvmTag = "llvmorg-21.1.8",
    [int]$ParallelLinkJobs = 6
)

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "env.ps1") -ToolchainRoot $ToolchainRoot

$src = Join-Path $ToolchainRoot "llvm-project"
$build = Join-Path $ToolchainRoot "llvm-build"
$install = Join-Path $ToolchainRoot "llvm-install"

if (-not (Test-Path $src)) {
    git clone --depth 1 --branch $LlvmTag -c core.autocrlf=false https://github.com/llvm/llvm-project.git $src
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
}

$cudaArgs = @("-DMLIR_ENABLE_CUDA_RUNNER=OFF")
if ($env:CUDA_PATH) {
    # nvcc only knows MSVC versions released before it; the CUDA runner itself is
    # plain C++ against the driver API, so a newer host compiler is fine.
    $env:NVCC_APPEND_FLAGS = "-allow-unsupported-compiler"
    $cudaPathFwd = $env:CUDA_PATH -replace "\\", "/"
    $cudaArgs = @(
        "-DMLIR_ENABLE_CUDA_RUNNER=ON",
        "-DCMAKE_CUDA_COMPILER=$cudaPathFwd/bin/nvcc.exe",
        "-DCMAKE_CUDA_ARCHITECTURES=86",
        "-DCUDAToolkit_ROOT=$cudaPathFwd"
    )
} else {
    Write-Warning "CUDA toolkit not found: building without mlir_cuda_runtime. Re-run after installing CUDA."
}

cmake -G Ninja -S (Join-Path $src "llvm") -B $build `
    -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_INSTALL_PREFIX=$install" `
    -DCMAKE_C_COMPILER=cl `
    -DCMAKE_CXX_COMPILER=cl `
    -DLLVM_ENABLE_PROJECTS=mlir `
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" `
    -DLLVM_ENABLE_ASSERTIONS=ON `
    -DLLVM_INSTALL_UTILS=ON `
    -DLLVM_INCLUDE_TESTS=OFF `
    -DLLVM_INCLUDE_BENCHMARKS=OFF `
    -DLLVM_INCLUDE_EXAMPLES=OFF `
    -DMLIR_INCLUDE_TESTS=OFF `
    "-DLLVM_PARALLEL_LINK_JOBS=$ParallelLinkJobs" `
    @cudaArgs
if ($LASTEXITCODE -ne 0) { throw "LLVM configure failed" }

cmake --build $build --target install
if ($LASTEXITCODE -ne 0) { throw "LLVM build failed" }

Write-Host "LLVM/MLIR installed to $install"
