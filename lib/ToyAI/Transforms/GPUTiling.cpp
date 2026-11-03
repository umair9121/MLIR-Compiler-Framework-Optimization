//===- GPUTiling.cpp - Tile scf.parallel into GPU blocks and threads ------===//
//
//   scf.parallel (%i, %j) = (0, 0) to (M, N)
// becomes
//   scf.parallel (%bi, %bj) = (0, 0) to (M, N) step (BY, BX)      // blocks
//     scf.parallel (%ti, %tj) = (0, 0) to (BY, BX)                // threads
//       scf.if (%bi + %ti < M && %bj + %tj < N) { ... }
//
// The last dimension (unit stride in row-major memrefs) is mapped to x so
// neighbouring threads of a warp touch neighbouring addresses (coalescing).
//
//===----------------------------------------------------------------------===//

#include "ToyAI/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/ParallelLoopMapper.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/IRMapping.h"

namespace mlir {
namespace toyai {
#define GEN_PASS_DEF_TOYAIGPUTILING
#include "ToyAI/Passes.h.inc"
} // namespace toyai
} // namespace mlir

using namespace mlir;

namespace {

/// Rewrites a reduction loop that does `x = load M[idx]; ...; store y, M[idx]`
/// with loop-invariant idx into an scf.for carrying the value in a register:
///   %init = load M[idx]
///   %r = scf.for ... iter_args(%acc = %init) { ...; yield %y }
///   store %r, M[idx]
/// Valid because M is not otherwise accessed in the loop and ToyAI buffers
/// never alias.
void promoteAccumulator(scf::ForOp loop) {
  if (loop.getNumRegionIterArgs() != 0)
    return;
  Block *body = loop.getBody();
  memref::LoadOp load;
  memref::StoreOp store;
  for (Operation &op : body->without_terminator()) {
    auto st = dyn_cast<memref::StoreOp>(op);
    if (!st)
      continue;
    for (Operation &prev : body->without_terminator()) {
      if (&prev == &op)
        break;
      auto ld = dyn_cast<memref::LoadOp>(prev);
      if (ld && ld.getMemRef() == st.getMemRef() &&
          ld.getIndices() == st.getIndices()) {
        load = ld;
        store = st;
      }
    }
    if (load)
      break;
  }
  if (!load ||
      !llvm::all_of(load.getIndices(), [&](Value v) {
        return loop.isDefinedOutsideOfLoop(v);
      }))
    return;
  bool otherAccess = false;
  loop.walk([&](Operation *op) {
    if (op == load || op == store)
      return;
    for (Value operand : op->getOperands())
      otherAccess |= operand == load.getMemRef();
  });
  if (otherAccess)
    return;

  OpBuilder b(loop);
  Location loc = loop.getLoc();
  Value init = b.create<memref::LoadOp>(loc, load.getMemRef(), load.getIndices());
  auto newLoop = b.create<scf::ForOp>(
      loc, loop.getLowerBound(), loop.getUpperBound(), loop.getStep(),
      ValueRange{init},
      [&](OpBuilder &nb, Location nloc, Value iv, ValueRange iters) {
        IRMapping map;
        map.map(loop.getInductionVar(), iv);
        map.map(load.getResult(), iters[0]);
        for (Operation &op : body->without_terminator())
          if (&op != load.getOperation() && &op != store.getOperation())
            nb.clone(op, map);
        nb.create<scf::YieldOp>(nloc,
                                map.lookupOrDefault(store.getValueToStore()));
      });
  b.create<memref::StoreOp>(loc, newLoop.getResult(0), store.getMemRef(),
                            store.getIndices());
  loop.erase();
}

gpu::ParallelLoopDimMappingAttr mapTo(MLIRContext *ctx, gpu::Processor p) {
  AffineMap id = AffineMap::getMultiDimIdentityMap(1, ctx);
  return gpu::ParallelLoopDimMappingAttr::get(ctx, p, id, id);
}

struct ToyAIGPUTilingPass
    : toyai::impl::ToyAIGPUTilingBase<ToyAIGPUTilingPass> {
  using Base::Base;

  void runOnOperation() override {
    SmallVector<scf::ParallelOp> roots;
    getOperation().walk([&](scf::ParallelOp op) {
      if (!op->getParentOfType<scf::ParallelOp>())
        roots.push_back(op);
    });

    for (scf::ParallelOp op : roots) {
      // Post-order: inner reductions are promoted first, which then exposes
      // the load/store pair to the enclosing loop.
      SmallVector<scf::ForOp> reductions;
      op.walk([&](scf::ForOp loop) { reductions.push_back(loop); });
      for (scf::ForOp loop : reductions)
        promoteAccumulator(loop);
    }

    MLIRContext *ctx = &getContext();
    for (scf::ParallelOp op : roots) {
      unsigned rank = op.getNumLoops();
      SmallVector<int64_t> tileSizes(rank, 1);
      if (rank == 1) {
        tileSizes[0] = blockX * blockY;
      } else {
        tileSizes[rank - 1] = blockX;
        tileSizes[rank - 2] = blockY;
      }

      auto [outer, inner] =
          scf::tileParallelLoop(op, tileSizes, /*noMinMaxBounds=*/true);

      static const gpu::Processor blocks[] = {
          gpu::Processor::BlockX, gpu::Processor::BlockY,
          gpu::Processor::BlockZ};
      static const gpu::Processor threads[] = {gpu::Processor::ThreadX,
                                               gpu::Processor::ThreadY};
      SmallVector<gpu::ParallelLoopDimMappingAttr> outerMap, innerMap;
      for (unsigned d = 0; d < rank; ++d) {
        unsigned fromInner = rank - 1 - d;
        outerMap.push_back(mapTo(ctx, fromInner < 3
                                          ? blocks[fromInner]
                                          : gpu::Processor::Sequential));
        bool threaded = rank == 1 ? fromInner == 0 : fromInner < 2;
        innerMap.push_back(mapTo(ctx, threaded ? threads[fromInner]
                                               : gpu::Processor::Sequential));
      }
      if (failed(gpu::setMappingAttr(outer, outerMap)) ||
          failed(gpu::setMappingAttr(inner, innerMap))) {
        outer.emitError("failed to set GPU mapping");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
