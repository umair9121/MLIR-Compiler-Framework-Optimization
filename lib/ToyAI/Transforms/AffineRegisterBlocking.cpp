//===- AffineRegisterBlocking.cpp - Unroll-and-jam around vector loops ----===//
//
// For matmul in i-k-j order the vectorized j loop does
//   C[i, j:j+8] += A[i, k] * B[k, j:j+8]
// Unroll-and-jamming k by U places U consecutive k updates of the same C
// vector next to each other; forwarding the stores to the following loads
// and dropping the overwritten stores means C is loaded and stored once per
// U multiply-adds instead of once per multiply-add. Unrolling the vector loop
// itself gives the out-of-order core independent accumulators to overlap.
//
// The forwarding is done here rather than left to affine-scalrep because
// scalrep must assume that distinct memref arguments may alias, which keeps
// every intermediate store of C alive.
//
//===----------------------------------------------------------------------===//

#include "ToyAI/Passes.h"

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace toyai {
#define GEN_PASS_DEF_TOYAIAFFINEREGISTERBLOCKING
#include "ToyAI/Passes.h.inc"
} // namespace toyai
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;

namespace {

uint64_t largestDivisorAtMost(uint64_t n, uint64_t bound) {
  for (uint64_t f = std::min(n, bound); f > 1; --f)
    if (n % f == 0)
      return f;
  return 1;
}

bool isVectorLoop(AffineForOp loop) {
  bool hasInnerLoop = false, hasVectorAccess = false;
  loop.getBody()->walk([&](Operation *op) {
    hasInnerLoop |= isa<AffineForOp>(op);
    hasVectorAccess |= isa<AffineVectorLoadOp, AffineVectorStoreOp>(op);
  });
  return !hasInnerLoop && hasVectorAccess;
}

/// Identifies an exact address: same memref value, map and map operands.
using AccessKey = std::tuple<Value, AffineMap, SmallVector<Value, 4>>;

template <typename OpTy>
AccessKey keyOf(OpTy op) {
  return {op.getMemRef(), op.getAffineMap(),
          SmallVector<Value, 4>(op.getMapOperands())};
}

/// Store-to-load forwarding and dead-store elimination inside one block.
///
/// ToyAI buffers never alias each other (every argument and every
/// allocation is a distinct array), so accesses to different memref values
/// are independent. Accesses to the same memref with a different address
/// are conservatively assumed to overlap.
void forwardAndEliminateStores(Block &block) {
  struct Pending {
    AccessKey key;
    AffineVectorStoreOp store;
    bool read;
  };
  SmallVector<Pending> pending;
  auto forgetMemref = [&](Value memref, const AccessKey *except) {
    llvm::erase_if(pending, [&](const Pending &p) {
      return std::get<0>(p.key) == memref && (!except || p.key != *except);
    });
  };

  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (auto load = dyn_cast<AffineVectorLoadOp>(op)) {
      AccessKey key = keyOf(load);
      auto it = llvm::find_if(pending, [&](Pending &p) {
        return p.key == key &&
               p.store.getValueToStore().getType() == load.getType();
      });
      if (it != pending.end()) {
        load.replaceAllUsesWith(it->store.getValueToStore());
        load.erase();
        continue;
      }
      for (Pending &p : pending)
        if (std::get<0>(p.key) == load.getMemRef())
          p.read = true;
    } else if (auto store = dyn_cast<AffineVectorStoreOp>(op)) {
      AccessKey key = keyOf(store);
      for (Pending &p : pending)
        if (p.key == key && !p.read)
          p.store.erase();
      forgetMemref(store.getMemRef(), nullptr);
      pending.push_back({key, store, false});
    } else if (auto load = dyn_cast<AffineLoadOp>(op)) {
      for (Pending &p : pending)
        if (std::get<0>(p.key) == load.getMemRef())
          p.read = true;
    } else if (auto store = dyn_cast<AffineStoreOp>(op)) {
      forgetMemref(store.getMemRef(), nullptr);
    } else if (!isMemoryEffectFree(&op)) {
      pending.clear();
    }
  }
}

SmallVector<AffineForOp> collectVectorLoops(func::FuncOp func) {
  SmallVector<AffineForOp> loops;
  func.walk([&](AffineForOp loop) {
    if (isVectorLoop(loop))
      loops.push_back(loop);
  });
  return loops;
}

struct ToyAIAffineRegisterBlockingPass
    : toyai::impl::ToyAIAffineRegisterBlockingBase<
          ToyAIAffineRegisterBlockingPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    if (unrollJamFactor > 1) {
      SmallVector<AffineForOp> parents;
      for (AffineForOp inner : collectVectorLoops(func)) {
        auto parent = dyn_cast<AffineForOp>(inner->getParentOp());
        // Only jam loops whose body is exactly the vector loop.
        if (!parent || parent.getBody()->getOperations().size() != 2)
          continue;
        parents.push_back(parent);
      }
      for (AffineForOp parent : parents) {
        std::optional<uint64_t> trip = getConstantTripCount(parent);
        if (!trip)
          continue;
        uint64_t factor = largestDivisorAtMost(*trip, unrollJamFactor);
        if (factor > 1)
          (void)loopUnrollJamByFactor(parent, factor);
      }
    }

    if (innerUnrollFactor > 1) {
      for (AffineForOp inner : collectVectorLoops(func)) {
        std::optional<uint64_t> trip = getConstantTripCount(inner);
        if (!trip)
          continue;
        uint64_t factor = largestDivisorAtMost(*trip, innerUnrollFactor);
        if (factor > 1)
          (void)loopUnrollByFactor(inner, factor);
      }
    }

    for (AffineForOp inner : collectVectorLoops(func))
      forwardAndEliminateStores(*inner.getBody());
  }
};

} // namespace
