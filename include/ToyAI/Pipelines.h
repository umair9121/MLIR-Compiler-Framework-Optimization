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

/// ToyAI -> linalg -> bufferized memrefs with results as out-params.
void buildToyAIBufferizationPipeline(OpPassManager &pm);

/// Full CPU lowering down to the LLVM dialect.
void buildToyAICpuPipeline(OpPassManager &pm, const CpuPipelineOptions &opts);

void registerToyAIPipelines();

} // namespace toyai
} // namespace mlir

#endif // TOYAI_PIPELINES_H
