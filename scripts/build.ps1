# Configures and builds the ToyAI project (toyai-opt + lit tests).
#
#   powershell -ExecutionPolicy Bypass -File scripts\build.ps1 [-Test]

param(
    [string]$ToolchainRoot = "D:\U\toolchains",
    [switch]$Test
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "env.ps1") -ToolchainRoot $ToolchainRoot

$repo = Split-Path -Parent $PSScriptRoot
$build = Join-Path $repo "build"
$lit = Join-Path $PSScriptRoot "run_lit.py"

cmake -G Ninja -S $repo -B $build `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_C_COMPILER=cl `
    -DCMAKE_CXX_COMPILER=cl `
    "-DMLIR_DIR=$env:MLIR_DIR" `
    "-DLLVM_EXTERNAL_LIT=$lit"
if ($LASTEXITCODE -ne 0) { throw "configure failed" }

cmake --build $build --target toyai-opt
if ($LASTEXITCODE -ne 0) { throw "build failed" }

if ($Test) {
    cmake --build $build --target check-toyai
    if ($LASTEXITCODE -ne 0) { throw "lit tests failed" }
}
