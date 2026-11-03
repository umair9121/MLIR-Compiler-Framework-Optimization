//===- Pipelines.cpp - End-to-end ToyAI lowering pipelines ----------------===//

#include "ToyAI/Pipelines.h"
#include "ToyAI/Passes.h"

#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
void addPipeline(OpPassManager &pm, StringRef text) {
  if (failed(parsePassPipeline(text, pm, llvm::errs())))
    llvm::report_fatal_error(
        llvm::Twine("toyai: invalid internal pass pipeline: ") + text);
}
} // namespace

void toyai::buildToyAIBufferizationPipeline(OpPassManager &pm) {
  addPipeline(pm,
              "convert-toyai-to-linalg,canonicalize,"
              "one-shot-bufferize{bufferize-function-boundaries "
              "function-boundary-type-conversion=identity-layout-map},"
              "buffer-results-to-out-params{hoist-static-allocs},"
              "func.func(toyai-memref-copy-to-linalg),canonicalize");
}

void toyai::buildToyAICpuPipeline(OpPassManager &pm,
                                  const CpuPipelineOptions &opts) {
  buildToyAIBufferizationPipeline(pm);
  addPipeline(pm, "buffer-deallocation-pipeline,"
                  "func.func(convert-linalg-to-affine-loops,canonicalize)");

  if (opts.optimize) {
    addPipeline(pm, llvm::formatv(
                        "func.func(toyai-affine-cache-tiling{{l1-kb={0} "
                        "l2-kb={1}},canonicalize,"
                        "toyai-affine-vectorize{{vector-width={2}},"
                        "toyai-affine-register-blocking{{unroll={3}},"
                        "affine-scalrep,canonicalize,"
                        "affine-loop-invariant-code-motion,canonicalize,cse)",
                        opts.l1KB.getValue(), opts.l2KB.getValue(),
                        opts.vectorWidth.getValue(), opts.unroll.getValue())
                        .str());
  }

  addPipeline(pm, "lower-affine,expand-strided-metadata,lower-affine,"
                  "convert-scf-to-cf,canonicalize,"
                  "convert-vector-to-llvm,finalize-memref-to-llvm,"
                  "convert-arith-to-llvm,convert-math-to-llvm,"
                  "convert-cf-to-llvm,convert-func-to-llvm,"
                  "convert-index-to-llvm,"
                  "reconcile-unrealized-casts,canonicalize,cse");
}

void toyai::buildToyAIGpuPipeline(OpPassManager &pm,
                                  const GpuPipelineOptions &opts) {
  buildToyAIBufferizationPipeline(pm);
  addPipeline(pm, llvm::formatv(
                      "func.func(convert-linalg-to-parallel-loops,canonicalize,"
                      "{2}"
                      "toyai-gpu-tiling{{block-x={0} block-y={1}},"
                      "toyai-gpu-data-movement,"
                      "convert-parallel-loops-to-gpu,canonicalize),"
                      "gpu-kernel-outlining,func.func(gpu-async-region)",
                      opts.blockX.getValue(), opts.blockY.getValue(),
                      opts.sharedMatmul ? "toyai-gpu-shared-matmul," : "")
                      .str());

  gpu::GPUToNVVMPipelineOptions nvvm;
  nvvm.cubinChip = opts.chip.getValue();
  nvvm.cubinFormat = opts.format.getValue();
  nvvm.optLevel = 3;
  gpu::buildLowerToNVVMPassPipeline(pm, nvvm);
}

void toyai::registerToyAIPipelines() {
  PassPipelineRegistration<>(
      "toyai-bufferize-pipeline",
      "Lower ToyAI to bufferized linalg with results as out-params",
      [](OpPassManager &pm) { buildToyAIBufferizationPipeline(pm); });
  PassPipelineRegistration<CpuPipelineOptions>(
      "toyai-cpu-pipeline",
      "Lower ToyAI to the LLVM dialect for x86 CPUs (tiled + vectorized)",
      buildToyAICpuPipeline);
  PassPipelineRegistration<GpuPipelineOptions>(
      "toyai-gpu-pipeline",
      "Lower ToyAI to NVVM kernels + host LLVM code for NVIDIA GPUs",
      buildToyAIGpuPipeline);
}
