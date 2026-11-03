//===- AffineCacheTiling.cpp - Cache-aware interchange + tiling -----------===//
//
// Works on the affine loop nests produced by convert-linalg-to-affine-loops.
//
// Interchange: the loop that indexes the fastest-varying (last) memref
// dimension of the most accesses is moved innermost, so the inner loop walks
// memory with unit stride. For matmul this turns i-j-k into i-k-j, where B and
// C are streamed row-wise and A[i, k] is loop invariant.
//
// Tiling: with the band's tileable loops named (from the inside out) j, k, i,
//   tj = min(trip_j, 128)                     -- long unit-stride inner rows
//   tk = L1/2 / tj                            -- a tk x tj panel of B fits L1
//   ti = L2/2 / (tk + tj)                     -- A and C panels fit L2
// A loop that fits its budget is left whole; otherwise the tile is rounded
// down to a power of two that divides the trip count, so the tiled nest has
// no partial tiles and constant inner trip counts.
//
//===----------------------------------------------------------------------===//

#include "AffineAccess.h"
#include "ToyAI/Passes.h"

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace toyai {
#define GEN_PASS_DEF_TOYAIAFFINECACHETILING
#include "ToyAI/Passes.h.inc"
} // namespace toyai
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::toyai;

namespace {

constexpr uint64_t kBytesPerElement = 4; // f32
constexpr uint64_t kMaxInnerTile = 128;

/// `trip` if it fits the budget, else the largest power of two that is
/// <= `budget` and divides `trip`.
uint64_t pickTile(uint64_t trip, uint64_t budget) {
  if (trip <= budget)
    return trip;
  uint64_t t = 1;
  while (t * 2 <= budget && trip % (t * 2) == 0)
    t *= 2;
  return t;
}

struct ToyAIAffineCacheTilingPass
    : toyai::impl::ToyAIAffineCacheTilingBase<ToyAIAffineCacheTilingPass> {
  using Base::Base;

  void runOnOperation() override {
    SmallVector<AffineForOp> roots;
    getOperation().walk([&](AffineForOp forOp) {
      if (!forOp->getParentOfType<AffineForOp>())
        roots.push_back(forOp);
    });
    for (AffineForOp root : roots)
      processBand(root);
  }

  /// Moves the loop with the most unit-stride accesses innermost. Returns the
  /// (possibly new) outermost loop of the band.
  AffineForOp interchange(SmallVectorImpl<AffineForOp> &band) {
    unsigned n = band.size();
    SmallVector<Operation *> accesses;
    band.back().walk([&](Operation *op) {
      if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op))
        accesses.push_back(op);
    });

    SmallVector<unsigned> scores(n, 0);
    for (unsigned i = 0; i < n; ++i)
      for (Operation *op : accesses)
        if (classifyAccess(op, band[i].getInductionVar()) ==
            AccessPattern::UnitStride)
          ++scores[i];

    unsigned best = n - 1;
    for (unsigned i = 0; i < n; ++i)
      if (scores[i] > scores[best])
        best = i;
    if (best == n - 1)
      return band.front();

    SmallVector<unsigned> perm(n);
    for (unsigned i = 0; i < n; ++i)
      perm[i] = i < best ? i : (i == best ? n - 1 : i - 1);
    if (!isValidLoopInterchangePermutation(band, perm))
      return band.front();

    unsigned newOuter = permuteLoops(band, perm);
    return band[newOuter];
  }

  void processBand(AffineForOp root) {
    SmallVector<AffineForOp> band;
    getPerfectlyNestedLoops(band, root);
    if (band.size() < 2)
      return;

    root = interchange(band);
    band.clear();
    getPerfectlyNestedLoops(band, root);

    unsigned n = band.size();
    SmallVector<uint64_t> trips;
    for (AffineForOp loop : band) {
      std::optional<uint64_t> trip = getConstantTripCount(loop);
      if (!trip || loop.getStepAsInt() != 1)
        return;
      trips.push_back(*trip);
    }

    // Roles from the inside out: j (unit stride), k, i.
    SmallVector<unsigned> roles;
    for (int i = n - 1; i >= 0 && roles.size() < 3; --i)
      if (trips[i] >= minTileTrip)
        roles.push_back(i);

    uint64_t l1Elems = l1KB * 1024 / kBytesPerElement / 2;
    uint64_t l2Elems = l2KB * 1024 / kBytesPerElement / 2;
    SmallVector<unsigned> sizes(trips.begin(), trips.end());
    if (roles.size() >= 1)
      sizes[roles[0]] = pickTile(trips[roles[0]], kMaxInnerTile);
    if (roles.size() >= 2)
      sizes[roles[1]] = pickTile(trips[roles[1]],
                                 std::max<uint64_t>(1, l1Elems / sizes[roles[0]]));
    if (roles.size() >= 3)
      sizes[roles[2]] = pickTile(
          trips[roles[2]],
          std::max<uint64_t>(1, l2Elems / (sizes[roles[1]] + sizes[roles[0]])));

    Attribute sizesAttr = DenseI64ArrayAttr::get(
        root.getContext(), SmallVector<int64_t>(sizes.begin(), sizes.end()));

    bool anyTiled = false;
    for (unsigned i = 0; i < n; ++i)
      anyTiled |= sizes[i] < trips[i];
    if (!anyTiled) {
      root->setAttr("toyai.tile_sizes", sizesAttr);
      return;
    }

    SmallVector<AffineForOp> tiled;
    if (failed(tilePerfectlyNested(band, sizes, &tiled)))
      return;
    // Tile-space loops whose tile covers the whole range run once.
    for (unsigned i = 0; i < n; ++i)
      if (sizes[i] == trips[i])
        (void)promoteIfSingleIteration(tiled[i]);

    AffineForOp outermost = tiled.back();
    while (auto parent = outermost->getParentOfType<AffineForOp>())
      outermost = parent;
    outermost->setAttr("toyai.tile_sizes", sizesAttr);
  }
};

} // namespace
