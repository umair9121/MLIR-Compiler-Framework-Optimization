#ifndef TOYAI_PIPELINES_H
#define TOYAI_PIPELINES_H

#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace toyai {

struct CpuPipelineOptions : public PassPipelineOptions<CpuPipelineOptions> {
  PassOptions::Option<bool> optimize{
      *this, "optimize",
      llvm::cl::desc("Run cache tiling, vectorization and register blocking"),
      llvm::cl::init(true)};
  PassOptions::Option<unsigned> l1KB{*this, "l1-kb",
                                     llvm::cl::desc("L1D size in KiB"),
                                     llvm::cl::init(48)};
  PassOptions::Option<unsigned> l2KB{*this, "l2-kb",
                                     llvm::cl::desc("L2 size in KiB"),
                                     llvm::cl::init(1280)};
  PassOptions::Option<unsigned> unroll{
      *this, "unroll", llvm::cl::desc("Max unroll-and-jam factor"),
      llvm::cl::init(4)};
  PassOptions::Option<unsigned> vectorWidth{
      *this, "vector-width", llvm::cl::desc("f32 lanes (0 disables)"),
      llvm::cl::init(8)};
};

struct GpuPipelineOptions : public PassPipelineOptions<GpuPipelineOptions> {
  PassOptions::Option<int64_t> blockX{*this, "block-x",
                                      llvm::cl::desc("Threads per block (x)"),
                                      llvm::cl::init(16)};
  PassOptions::Option<int64_t> blockY{*this, "block-y",
                                      llvm::cl::desc("Threads per block (y)"),
                                      llvm::cl::init(16)};
  PassOptions::Option<bool> sharedMatmul{
      *this, "shared-matmul",
      llvm::cl::desc("Use the shared-memory tiled kernel for matmul"),
      llvm::cl::init(true)};
  PassOptions::Option<std::string> chip{*this, "chip",
                                        llvm::cl::desc("NVIDIA SM target"),
                                        llvm::cl::init("sm_86")};
  PassOptions::Option<std::string> format{
      *this, "format",
      llvm::cl::desc("GPU binary format: fatbin, bin (cubin) or isa (PTX)"),
      llvm::cl::init("fatbin")};
};

/// ToyAI -> linalg -> bufferized memrefs with results as out-params.
void buildToyAIBufferizationPipeline(OpPassManager &pm);

/// Full CPU lowering down to the LLVM dialect.
void buildToyAICpuPipeline(OpPassManager &pm, const CpuPipelineOptions &opts);

/// Full NVIDIA GPU lowering down to the LLVM dialect + embedded GPU binary.
void buildToyAIGpuPipeline(OpPassManager &pm, const GpuPipelineOptions &opts);

void registerToyAIPipelines();

} // namespace toyai
} // namespace mlir

#endif // TOYAI_PIPELINES_H
