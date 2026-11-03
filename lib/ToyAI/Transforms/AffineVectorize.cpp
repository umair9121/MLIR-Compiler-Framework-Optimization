//===- AffineVectorize.cpp - Innermost-loop vectorization -----------------===//
//
// A deliberately small loop vectorizer that stays in the affine dialect:
//
//   affine.for %j = 0 to 128 {                affine.for %j = 0 to 128 step 8 {
//     %a = affine.load %A[%i, %k]               %a  = affine.load %A[%i, %k]
//     %b = affine.load %B[%k, %j]       ==>     %av = vector.broadcast %a
//     %c = affine.load %C[%i, %j]               %b  = affine.vector_load %B[%k, %j]
//     ...arith...                               %c  = affine.vector_load %C[%i, %j]
//     affine.store %r, %C[%i, %j]               ...arith on vector<8xf32>...
//   }                                           affine.vector_store %r, %C[%i, %j]
//                                             }
//
// Because the result is still affine, affine-scalrep and unroll-and-jam can
// keep working on it (the upstream super-vectorizer emits vector.transfer ops,
// which those passes do not understand).
//
//===----------------------------------------------------------------------===//

#include "AffineAccess.h"
#include "ToyAI/Passes.h"

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"

namespace mlir {
namespace toyai {
#define GEN_PASS_DEF_TOYAIAFFINEVECTORIZE
#include "ToyAI/Passes.h.inc"
} // namespace toyai
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::toyai;

namespace {

bool isElementwiseFloatOp(Operation *op) {
  return isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
             arith::MaximumFOp, arith::MinimumFOp, arith::MaxNumFOp,
             arith::MinNumFOp, arith::NegFOp>(op);
}

/// Checks that every op in the loop body can be vectorized along `iv` and the
/// loop runs at least one full vector.
bool canVectorize(AffineForOp loop, unsigned width) {
  if (loop.getNumResults() != 0 || loop.getStepAsInt() != 1)
    return false;
  std::optional<uint64_t> trip = getConstantTripCount(loop);
  if (!trip || *trip < width)
    return false;

  Value iv = loop.getInductionVar();
  bool sawVectorAccess = false;
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (auto load = dyn_cast<AffineLoadOp>(op)) {
      if (!load.getType().isF32())
        return false;
      AccessPattern p = classifyAccess(load, iv);
      if (p == AccessPattern::Other)
        return false;
      sawVectorAccess |= p == AccessPattern::UnitStride;
    } else if (auto store = dyn_cast<AffineStoreOp>(op)) {
      // An invariant store inside the loop would be a reduction.
      if (classifyAccess(store, iv) != AccessPattern::UnitStride ||
          !store.getValueToStore().getType().isF32())
        return false;
      sawVectorAccess = true;
    } else if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
      if (!cst.getType().isF32())
        return false;
    } else if (!isElementwiseFloatOp(&op)) {
      return false;
    }
  }
  return sawVectorAccess;
}

/// If the trip count is not a multiple of `width`, splits off the last
/// `trip % width` iterations into a scalar epilogue loop. Returns false if the
/// loop bounds are too complex to split.
bool peelRemainder(AffineForOp loop, unsigned width) {
  uint64_t trip = *getConstantTripCount(loop);
  uint64_t remainder = trip % width;
  if (remainder == 0)
    return true;
  AffineMap lbMap = loop.getLowerBoundMap();
  if (lbMap.getNumResults() != 1)
    return false;
  AffineMap splitMap =
      AffineMap::get(lbMap.getNumDims(), lbMap.getNumSymbols(),
                     lbMap.getResult(0) + (trip - remainder));
  SmallVector<Value> lbOperands(loop.getLowerBoundOperands());

  OpBuilder b(loop);
  b.setInsertionPointAfter(loop);
  auto epilogue = cast<AffineForOp>(b.clone(*loop.getOperation()));
  epilogue.setLowerBound(lbOperands, splitMap);
  loop.setUpperBound(lbOperands, splitMap);
  return true;
}

void vectorize(AffineForOp loop, unsigned width) {
  auto vecType = VectorType::get({width}, Float32Type::get(loop.getContext()));
  Value iv = loop.getInductionVar();
  IRMapping vectorOf;
  SmallVector<Operation *> toErase;
  OpBuilder b(loop.getContext());

  // Scalars defined outside the loop are broadcast once, before it.
  auto vectorValue = [&](Value v) -> Value {
    if (Value mapped = vectorOf.lookupOrNull(v))
      return mapped;
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(loop);
    Value splat = b.create<vector::BroadcastOp>(v.getLoc(), vecType, v);
    vectorOf.map(v, splat);
    return splat;
  };

  for (Operation &op : llvm::make_early_inc_range(
           loop.getBody()->without_terminator())) {
    b.setInsertionPoint(&op);
    Location loc = op.getLoc();
    if (auto load = dyn_cast<AffineLoadOp>(op)) {
      if (classifyAccess(load, iv) == AccessPattern::Invariant) {
        b.setInsertionPointAfter(load);
        vectorOf.map(load.getResult(),
                     b.create<vector::BroadcastOp>(loc, vecType, load));
        continue;
      }
      Value v = b.create<AffineVectorLoadOp>(loc, vecType, load.getMemRef(),
                                             load.getAffineMap(),
                                             load.getMapOperands());
      vectorOf.map(load.getResult(), v);
      toErase.push_back(load);
    } else if (auto store = dyn_cast<AffineStoreOp>(op)) {
      b.create<AffineVectorStoreOp>(loc, vectorValue(store.getValueToStore()),
                                    store.getMemRef(), store.getAffineMap(),
                                    store.getMapOperands());
      toErase.push_back(store);
    } else if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
      auto splat = DenseElementsAttr::get(vecType, cst.getValue());
      vectorOf.map(cst.getResult(), b.create<arith::ConstantOp>(loc, splat));
      toErase.push_back(cst);
    } else {
      SmallVector<Value> operands;
      for (Value operand : op.getOperands())
        operands.push_back(vectorValue(operand));
      OperationState state(loc, op.getName().getStringRef(), operands,
                           TypeRange{vecType}, op.getAttrs());
      Operation *vecOp = b.create(state);
      vectorOf.map(op.getResult(0), vecOp->getResult(0));
      toErase.push_back(&op);
    }
  }

  for (Operation *op : llvm::reverse(toErase))
    op->erase();
  loop.setStep(width);
}

struct ToyAIAffineVectorizePass
    : toyai::impl::ToyAIAffineVectorizeBase<ToyAIAffineVectorizePass> {
  using Base::Base;

  void runOnOperation() override {
    if (vectorWidth < 2)
      return;
    SmallVector<AffineForOp> innermost;
    getOperation().walk([&](AffineForOp loop) {
      bool hasInnerLoop = false;
      loop.getBody()->walk([&](AffineForOp) { hasInnerLoop = true; });
      if (!hasInnerLoop && canVectorize(loop, vectorWidth))
        innermost.push_back(loop);
    });
    for (AffineForOp loop : innermost)
      if (peelRemainder(loop, vectorWidth))
        vectorize(loop, vectorWidth);
  }
};

} // namespace
