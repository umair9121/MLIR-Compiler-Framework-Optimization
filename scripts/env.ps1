# Dot-source this script to get a shell that can build and run the ToyAI pipeline:
#   . .\scripts\env.ps1
#
# It imports the MSVC x64 developer environment, then puts the installed
# LLVM/MLIR toolchain, the CUDA toolkit and the project build tree on PATH.

param(
    [string]$ToolchainRoot = $(if ($env:TOYAI_TOOLCHAIN_ROOT) { $env:TOYAI_TOOLCHAIN_ROOT } else { "D:\U\toolchains" })
)

$ErrorActionPreference = "Stop"

function Import-VsDevEnv {
    if ($env:VSCMD_ARG_TGT_ARCH -eq "x64") { return }
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $vcvars = & $vswhere -all -products * -property installationPath |
        ForEach-Object { Join-Path $_ "VC\Auxiliary\Build\vcvars64.bat" } |
        Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $vcvars) { throw "No Visual Studio installation with the MSVC x64 toolset was found." }
    $lines = cmd /c "`"$vcvars`" >nul 2>&1 && set"
    foreach ($line in $lines) {
        $idx = $line.IndexOf("=")
        if ($idx -gt 0) {
            Set-Item -Path ("Env:" + $line.Substring(0, $idx)) -Value $line.Substring($idx + 1)
        }
    }
}

Import-VsDevEnv

$repoRoot = Split-Path -Parent $PSScriptRoot
$llvmInstall = Join-Path $ToolchainRoot "llvm-install"

$env:TOYAI_ROOT = $repoRoot
$env:TOYAI_LLVM_INSTALL = $llvmInstall
$env:MLIR_DIR = Join-Path $llvmInstall "lib\cmake\mlir"
$env:LLVM_DIR = Join-Path $llvmInstall "lib\cmake\llvm"

if (-not $env:CUDA_PATH) {
    $cuda = Get-ChildItem "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | Select-Object -First 1
    if ($cuda) { $env:CUDA_PATH = $cuda.FullName }
}

$pathParts = @(
    (Join-Path $repoRoot "build\bin"),
    (Join-Path $llvmInstall "bin")
)
if ($env:CUDA_PATH) { $pathParts += (Join-Path $env:CUDA_PATH "bin") }
$env:PATH = ($pathParts -join ";") + ";" + $env:PATH

Write-Host "ToyAI environment ready"
Write-Host "  repo       : $repoRoot"
Write-Host "  llvm/mlir  : $llvmInstall"
Write-Host "  cuda       : $env:CUDA_PATH"
