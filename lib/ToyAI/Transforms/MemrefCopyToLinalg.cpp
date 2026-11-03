//===- MemrefCopyToLinalg.cpp - memref.copy -> linalg.copy ----------------===//
//
// Bufferization materializes tensor.pad / insert_slice as memref.copy into a
// strided subview. Lowered directly, that becomes a call into the MLIR runtime
// on the host. As linalg.copy it is just another loop nest, so it gets tiled
// on CPU and turned into a kernel on GPU like everything else.
//
//===----------------------------------------------------------------------===//

#include "ToyAI/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace toyai {
#define GEN_PASS_DEF_TOYAIMEMREFCOPYTOLINALG
#include "ToyAI/Passes.h.inc"
} // namespace toyai
} // namespace mlir

using namespace mlir;

namespace {
struct ToyAIMemrefCopyToLinalgPass
    : toyai::impl::ToyAIMemrefCopyToLinalgBase<ToyAIMemrefCopyToLinalgPass> {
  void runOnOperation() override {
    SmallVector<memref::CopyOp> copies;
    getOperation().walk([&](memref::CopyOp op) { copies.push_back(op); });
    IRRewriter rewriter(&getContext());
    for (memref::CopyOp copy : copies) {
      rewriter.setInsertionPoint(copy);
      rewriter.replaceOpWithNewOp<linalg::CopyOp>(
          copy, ValueRange{copy.getSource()}, ValueRange{copy.getTarget()});
    }
  }
};
} // namespace
