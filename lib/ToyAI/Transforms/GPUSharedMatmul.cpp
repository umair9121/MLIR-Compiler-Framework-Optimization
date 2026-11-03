//===- GPUSharedMatmul.cpp - Shared-memory tiled matmul kernel ------------===//
//
// Matches the loop nest that linalg.matmul becomes after
// convert-linalg-to-parallel-loops,
//
//   scf.parallel (%i, %j) = (0, 0) to (M, N) {
//     scf.for %k = 0 to K {
//       C[i, j] += A[i, k] * B[k, j]
//     }
//   }
//
// and replaces it with a gpu.launch that computes a 64x64 tile of C per
// 16x16-thread block:
//   * each thread owns a 4x4 register tile of C (rows ty + 16r, cols tx + 16c,
//     so neighbouring threads touch neighbouring columns: coalesced global
//     accesses, conflict-free shared-memory accesses),
//   * K is walked in steps of 16; every step the block cooperatively stages a
//     64x16 panel of A and a 16x64 panel of B in shared (workgroup) memory,
//     synchronizes, and does 16 rank-1 updates of its register tile from it.
// Each element of A and B is read from global memory 64x less often than in
// the one-thread-per-output kernel.
//
// Shapes that are not multiples of the tile sizes are left alone and go
// through the generic toyai-gpu-tiling path.
//
//===----------------------------------------------------------------------===//

#include "ToyAI/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace toyai {
#define GEN_PASS_DEF_TOYAIGPUSHAREDMATMUL
#include "ToyAI/Passes.h.inc"
} // namespace toyai
} // namespace mlir

using namespace mlir;

namespace {

constexpr int64_t kThreads = 16;          // threads per block, x and y
constexpr int64_t kRegTile = 4;           // outputs per thread, per dim
constexpr int64_t kTileMN = kThreads * kRegTile; // 64
constexpr int64_t kTileK = 16;

struct MatmulNest {
  scf::ParallelOp par;
  Value A, B, C;
  int64_t M, N, K;
};

std::optional<int64_t> constIndex(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantIndexOp>())
    return c.value();
  return std::nullopt;
}

bool indicesAre(memref::LoadOp op, Value a, Value b) {
  return op.getIndices().size() == 2 && op.getIndices()[0] == a &&
         op.getIndices()[1] == b;
}

std::optional<MatmulNest> matchMatmul(scf::ParallelOp par) {
  if (par.getNumLoops() != 2 || par.getNumResults() != 0)
    return std::nullopt;
  for (Value lb : par.getLowerBound())
    if (constIndex(lb) != 0)
      return std::nullopt;
  for (Value st : par.getStep())
    if (constIndex(st) != 1)
      return std::nullopt;
  auto m = constIndex(par.getUpperBound()[0]);
  auto n = constIndex(par.getUpperBound()[1]);
  if (!m || !n)
    return std::nullopt;

  Block *pbody = par.getBody();
  if (pbody->getOperations().size() != 2)
    return std::nullopt;
  auto loop = dyn_cast<scf::ForOp>(pbody->front());
  if (!loop || loop.getNumRegionIterArgs() != 0 ||
      constIndex(loop.getLowerBound()) != 0 || constIndex(loop.getStep()) != 1)
    return std::nullopt;
  auto k = constIndex(loop.getUpperBound());
  if (!k)
    return std::nullopt;

  Value i = par.getInductionVars()[0], j = par.getInductionVars()[1];
  Value kv = loop.getInductionVar();
  memref::LoadOp la, lb, lc;
  arith::MulFOp mul;
  arith::AddFOp add;
  memref::StoreOp store;
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (auto ld = dyn_cast<memref::LoadOp>(op)) {
      if (indicesAre(ld, i, kv))
        la = ld;
      else if (indicesAre(ld, kv, j))
        lb = ld;
      else if (indicesAre(ld, i, j))
        lc = ld;
      else
        return std::nullopt;
    } else if (auto m = dyn_cast<arith::MulFOp>(op)) {
      mul = m;
    } else if (auto a = dyn_cast<arith::AddFOp>(op)) {
      add = a;
    } else if (auto s = dyn_cast<memref::StoreOp>(op)) {
      store = s;
    } else {
      return std::nullopt;
    }
  }
  if (!la || !lb || !lc || !mul || !add || !store)
    return std::nullopt;
  if (loop.getBody()->getOperations().size() != 7)
    return std::nullopt;
  auto isPair = [](Operation *op, Value x, Value y) {
    return (op->getOperand(0) == x && op->getOperand(1) == y) ||
           (op->getOperand(0) == y && op->getOperand(1) == x);
  };
  if (!isPair(mul, la, lb) || !isPair(add, lc, mul) ||
      store.getValueToStore() != add.getResult() ||
      store.getMemRef() != lc.getMemRef() ||
      store.getIndices() != lc.getIndices())
    return std::nullopt;
  if (!la.getType().isF32())
    return std::nullopt;

  return MatmulNest{par, la.getMemRef(), lb.getMemRef(), lc.getMemRef(),
                    *m, *n, *k};
}

void emitSharedMemoryKernel(const MatmulNest &mm) {
  scf::ParallelOp par = mm.par;
  Location loc = par.getLoc();
  MLIRContext *ctx = par.getContext();
  OpBuilder b(par);
  Type f32 = b.getF32Type();

  auto cst = [&](int64_t v) -> Value {
    return b.create<arith::ConstantIndexOp>(loc, v);
  };
  Value c1 = cst(1);
  auto workgroup = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
  auto aTileType = MemRefType::get({kTileMN, kTileK}, f32, AffineMap(), workgroup);
  auto bTileType = MemRefType::get({kTileK, kTileMN}, f32, AffineMap(), workgroup);

  auto launch = b.create<gpu::LaunchOp>(
      loc, /*grid=*/cst(mm.N / kTileMN), cst(mm.M / kTileMN), c1,
      /*block=*/cst(kThreads), cst(kThreads), c1,
      /*dynamicSharedMemorySize=*/nullptr, /*asyncTokenType=*/nullptr,
      /*asyncDependencies=*/ValueRange{},
      /*workgroupAttributions=*/TypeRange{aTileType, bTileType});

  Block &body = launch.getBody().front();
  b.setInsertionPointToStart(&body);
  Value aTile = launch.getWorkgroupAttributions()[0];
  Value bTile = launch.getWorkgroupAttributions()[1];
  Value tx = launch.getThreadIds().x, ty = launch.getThreadIds().y;
  Value rowBase = b.create<arith::MulIOp>(loc, launch.getBlockIds().y, cst(kTileMN));
  Value colBase = b.create<arith::MulIOp>(loc, launch.getBlockIds().x, cst(kTileMN));

  // Thread-local offsets inside the tile: ty + 16r and tx + 16c.
  SmallVector<Value> rowOff, colOff, rows, cols;
  for (int64_t r = 0; r < kRegTile; ++r) {
    rowOff.push_back(b.create<arith::AddIOp>(loc, ty, cst(r * kThreads)));
    colOff.push_back(b.create<arith::AddIOp>(loc, tx, cst(r * kThreads)));
    rows.push_back(b.create<arith::AddIOp>(loc, rowBase, rowOff.back()));
    cols.push_back(b.create<arith::AddIOp>(loc, colBase, colOff.back()));
  }

  SmallVector<Value> init;
  for (int64_t r = 0; r < kRegTile; ++r)
    for (int64_t c = 0; c < kRegTile; ++c)
      init.push_back(b.create<memref::LoadOp>(loc, mm.C, ValueRange{rows[r], cols[c]}));

  auto kLoop = b.create<scf::ForOp>(
      loc, cst(0), cst(mm.K), cst(kTileK), init,
      [&](OpBuilder &kb, Location kl, Value k0, ValueRange acc) {
        // Stage A[rowBase:+64, k0:+16] and B[k0:+16, colBase:+64].
        Value aCol = kb.create<arith::AddIOp>(kl, k0, tx);
        Value bRow = kb.create<arith::AddIOp>(kl, k0, ty);
        for (int64_t r = 0; r < kRegTile; ++r) {
          Value a = kb.create<memref::LoadOp>(kl, mm.A, ValueRange{rows[r], aCol});
          kb.create<memref::StoreOp>(kl, a, aTile, ValueRange{rowOff[r], tx});
          Value bv = kb.create<memref::LoadOp>(kl, mm.B, ValueRange{bRow, cols[r]});
          kb.create<memref::StoreOp>(kl, bv, bTile, ValueRange{ty, colOff[r]});
        }
        kb.create<gpu::BarrierOp>(kl);

        auto inner = kb.create<scf::ForOp>(
            kl, kb.create<arith::ConstantIndexOp>(kl, 0),
            kb.create<arith::ConstantIndexOp>(kl, kTileK),
            kb.create<arith::ConstantIndexOp>(kl, 1), acc,
            [&](OpBuilder &ib, Location il, Value kk, ValueRange acc2) {
              SmallVector<Value> av, bv, next;
              for (int64_t r = 0; r < kRegTile; ++r) {
                av.push_back(ib.create<memref::LoadOp>(il, aTile, ValueRange{rowOff[r], kk}));
                bv.push_back(ib.create<memref::LoadOp>(il, bTile, ValueRange{kk, colOff[r]}));
              }
              for (int64_t r = 0; r < kRegTile; ++r)
                for (int64_t c = 0; c < kRegTile; ++c) {
                  Value p = ib.create<arith::MulFOp>(il, av[r], bv[c]);
                  next.push_back(ib.create<arith::AddFOp>(il, acc2[r * kRegTile + c], p));
                }
              ib.create<scf::YieldOp>(il, next);
            });
        kb.create<gpu::BarrierOp>(kl);
        kb.create<scf::YieldOp>(kl, inner.getResults());
      });

  for (int64_t r = 0; r < kRegTile; ++r)
    for (int64_t c = 0; c < kRegTile; ++c)
      b.create<memref::StoreOp>(loc, kLoop.getResult(r * kRegTile + c), mm.C,
                                ValueRange{rows[r], cols[c]});
  b.create<gpu::TerminatorOp>(loc);
  par.erase();
}

struct ToyAIGPUSharedMatmulPass
    : toyai::impl::ToyAIGPUSharedMatmulBase<ToyAIGPUSharedMatmulPass> {
  void runOnOperation() override {
    SmallVector<MatmulNest> nests;
    getOperation().walk([&](scf::ParallelOp par) {
      if (par->getParentOfType<scf::ParallelOp>())
        return;
      if (auto mm = matchMatmul(par))
        if (mm->M % kTileMN == 0 && mm->N % kTileMN == 0 && mm->K % kTileK == 0)
          nests.push_back(*mm);
    });
    for (const MatmulNest &mm : nests)
      emitSharedMemoryKernel(mm);
  }
};

} // namespace
