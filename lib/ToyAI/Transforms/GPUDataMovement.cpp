//===- GPUDataMovement.cpp - Device buffers + host<->device copies --------===//
//
// Runs after linalg has been lowered to (GPU-mapped) scf.parallel loops, when
// every memory access in the function is a memref.load / memref.store inside
// a loop that will become a kernel. The pass:
//   * gives every memref argument a device twin (gpu.alloc),
//   * uploads it on entry unless the first op touching it only writes it,
//   * downloads it before returning if anything writes it,
//   * turns function-local memref.alloc into gpu.alloc,
//   * frees all device buffers before returning.
// The resulting synchronous gpu ops are made async by gpu-async-region.
//
//===----------------------------------------------------------------------===//

#include "ToyAI/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
namespace toyai {
#define GEN_PASS_DEF_TOYAIGPUDATAMOVEMENT
#include "ToyAI/Passes.h.inc"
} // namespace toyai
} // namespace mlir

using namespace mlir;

namespace {

struct BufferUse {
  bool read = false;
  bool written = false;
  bool needsUpload = false;
};

/// The value plus every view derived from it (subview, cast, ...).
llvm::SetVector<Value> collectAliases(Value root) {
  llvm::SetVector<Value> aliases;
  aliases.insert(root);
  for (unsigned i = 0; i < aliases.size(); ++i)
    for (Operation *user : aliases[i].getUsers())
      if (auto view = dyn_cast<ViewLikeOpInterface>(user))
        if (view.getViewSource() == aliases[i])
          for (Value r : user->getResults())
            aliases.insert(r);
  return aliases;
}

BufferUse analyze(Value buffer, Block &body) {
  llvm::SetVector<Value> aliases = collectAliases(buffer);
  BufferUse use;
  bool firstTouchSeen = false;
  for (Operation &top : body.without_terminator()) {
    bool reads = false, writes = false;
    top.walk([&](Operation *op) {
      if (auto load = dyn_cast<memref::LoadOp>(op)) {
        reads |= aliases.contains(load.getMemRef());
      } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
        writes |= aliases.contains(store.getMemRef());
      } else if (!isa<ViewLikeOpInterface>(op)) {
        // Unknown user: assume it both reads and writes.
        for (Value operand : op->getOperands())
          if (aliases.contains(operand))
            reads = writes = true;
      }
    });
    if ((reads || writes) && !firstTouchSeen) {
      firstTouchSeen = true;
      use.needsUpload = reads;
    }
    use.read |= reads;
    use.written |= writes;
  }
  return use;
}

struct ToyAIGPUDataMovementPass
    : toyai::impl::ToyAIGPUDataMovementBase<ToyAIGPUDataMovementPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;
    if (!func.getBody().hasOneBlock()) {
      func.emitError("toyai-gpu-data-movement expects a single-block function");
      return signalPassFailure();
    }
    Block &body = func.getBody().front();
    auto ret = cast<func::ReturnOp>(body.getTerminator());
    Location loc = func.getLoc();
    OpBuilder b(&getContext());

    auto deviceAlloc = [&](MemRefType type) -> Value {
      return b
          .create<gpu::AllocOp>(loc, type, /*asyncToken=*/Type(),
                                /*asyncDependencies=*/ValueRange{},
                                /*dynamicSizes=*/ValueRange{},
                                /*symbolOperands=*/ValueRange{},
                                /*hostShared=*/false)
          .getMemref();
    };
    auto memcpy = [&](Value dst, Value src) {
      b.create<gpu::MemcpyOp>(loc, /*asyncToken=*/Type(), ValueRange{}, dst,
                              src);
    };

    SmallVector<Value> deviceBuffers;

    // Local buffers live on the device only.
    SmallVector<memref::AllocOp> allocs;
    func.walk([&](memref::AllocOp a) { allocs.push_back(a); });
    for (memref::AllocOp alloc : allocs) {
      if (!alloc.getDynamicSizes().empty()) {
        alloc.emitError("dynamic allocations are not supported on the GPU");
        return signalPassFailure();
      }
      b.setInsertionPoint(alloc);
      Value dev = deviceAlloc(alloc.getType());
      alloc.replaceAllUsesWith(dev);
      alloc.erase();
      deviceBuffers.push_back(dev);
    }
    SmallVector<memref::DeallocOp> deallocs;
    func.walk([&](memref::DeallocOp d) { deallocs.push_back(d); });
    for (memref::DeallocOp d : deallocs)
      d.erase();

    // Arguments get a device twin plus the copies they need.
    SmallVector<std::pair<Value, Value>> downloads; // (host, device)
    b.setInsertionPointToStart(&body);
    for (BlockArgument arg : func.getArguments()) {
      auto type = dyn_cast<MemRefType>(arg.getType());
      if (!type)
        continue;
      BufferUse use = analyze(arg, body);
      if (!use.read && !use.written)
        continue;
      Value dev = deviceAlloc(type);
      arg.replaceAllUsesWith(dev);
      if (use.needsUpload)
        memcpy(dev, arg);
      if (use.written)
        downloads.push_back({arg, dev});
      deviceBuffers.push_back(dev);
    }

    b.setInsertionPoint(ret);
    for (auto [host, dev] : downloads)
      memcpy(host, dev);
    for (Value dev : deviceBuffers)
      b.create<gpu::DeallocOp>(loc, /*asyncToken=*/Type(), ValueRange{}, dev);
  }
};

} // namespace
