//===- AffineAccess.h - Stride analysis of affine memory accesses ---------===//

#ifndef TOYAI_TRANSFORMS_AFFINEACCESS_H
#define TOYAI_TRANSFORMS_AFFINEACCESS_H

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"

namespace mlir {
namespace toyai {

/// How a memory access moves when the loop induction variable `iv` advances
/// by one iteration.
enum class AccessPattern {
  Invariant,  // address does not depend on iv
  UnitStride, // only the last subscript depends on iv, with coefficient 1
  Other,      // anything else (strided, multi-dim, non-affine)
};

inline AccessPattern classifyAccess(Operation *memOp, Value iv) {
  affine::MemRefAccess access(memOp);
  affine::AffineValueMap vmap;
  access.getAccessMap(&vmap);
  AffineMap map = vmap.getAffineMap();

  std::optional<unsigned> pos;
  for (auto [i, operand] : llvm::enumerate(vmap.getOperands()))
    if (operand == iv)
      pos = i;
  if (!pos)
    return AccessPattern::Invariant;

  unsigned numDims = map.getNumDims();
  MLIRContext *ctx = map.getContext();
  AffineExpr var = *pos < numDims
                       ? getAffineDimExpr(*pos, ctx)
                       : getAffineSymbolExpr(*pos - numDims, ctx);

  unsigned rank = map.getNumResults();
  bool dependsOnIv = false;
  for (unsigned r = 0; r < rank; ++r) {
    AffineExpr e = map.getResult(r);
    bool uses = *pos < numDims ? e.isFunctionOfDim(*pos)
                               : e.isFunctionOfSymbol(*pos - numDims);
    if (!uses)
      continue;
    dependsOnIv = true;
    if (r != rank - 1)
      return AccessPattern::Other;
    AffineExpr delta = simplifyAffineExpr(e.replace(var, var + 1) - e, numDims,
                                          map.getNumSymbols());
    auto cst = dyn_cast<AffineConstantExpr>(delta);
    if (!cst || cst.getValue() != 1)
      return AccessPattern::Other;
  }
  return dependsOnIv ? AccessPattern::UnitStride : AccessPattern::Invariant;
}

} // namespace toyai
} // namespace mlir

#endif // TOYAI_TRANSFORMS_AFFINEACCESS_H
