#ifndef TOYAI_PASSES_TD
#define TOYAI_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertToyAIToLinalg : Pass<"convert-toyai-to-linalg", "ModuleOp"> {
  let summary = "Lower ToyAI ops to linalg on tensors";
  let description = [{
    matmul  -> tensor.empty + linalg.fill + linalg.matmul
    conv2d  -> tensor.pad (if padding) + linalg.conv_2d_nchw_fchw
    add     -> elementwise linalg.generic (addf)
    relu    -> elementwise linalg.generic (maximumf with 0)
    constant-> tensor.empty + linalg.fill
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::tensor::TensorDialect",
  ];
}

def ToyAIMemrefCopyToLinalg : Pass<"toyai-memref-copy-to-linalg", "func::FuncOp"> {
  let summary = "Rewrite memref.copy into linalg.copy so copies become loops/kernels";
  let dependentDialects = ["::mlir::linalg::LinalgDialect"];
}

def ToyAIAffineCacheTiling : Pass<"toyai-affine-cache-tiling", "func::FuncOp"> {
  let summary = "Cache-aware loop interchange and tiling of affine loop nests";
  let description = [{
    For every perfectly nested affine band:
      1. Interchange so the loop indexing the fastest-varying memref dimension
         of the most accesses becomes innermost (unit-stride inner loop),
         if the permutation is legal.
      2. Tile the band so that the inner working set fits in L1 and the
         outer working set fits in L2. Tile sizes are powers of two that
         divide the trip counts, so no partial tiles are generated.
  }];
  let options = [
    Option<"l1KB", "l1-kb", "unsigned", /*default=*/"48", "L1 data cache size in KiB">,
    Option<"l2KB", "l2-kb", "unsigned", /*default=*/"1280", "L2 cache size in KiB">,
    Option<"minTileTrip", "min-trip", "unsigned", /*default=*/"16",
           "Loops with fewer iterations than this are not tiled">,
  ];
  let dependentDialects = ["::mlir::affine::AffineDialect"];
}

def ToyAIAffineVectorize : Pass<"toyai-affine-vectorize", "func::FuncOp"> {
  let summary = "Vectorize innermost unit-stride affine loops";
  let description = [{
    Rewrites an innermost `affine.for` with a constant trip count divisible by
    the vector width into a loop of step `vector-width` whose loads and stores
    are `affine.vector_load` / `affine.vector_store`. Loads invariant in the
    loop become scalar loads + `vector.broadcast`. Loops containing anything
    other than affine loads/stores and elementwise float arithmetic, or with
    non-unit-stride accesses, are left untouched.
  }];
  let options = [
    Option<"vectorWidth", "vector-width", "unsigned", /*default=*/"8",
           "Number of f32 lanes (8 = AVX2)">,
  ];
  let dependentDialects = ["::mlir::vector::VectorDialect"];
}

def ToyAIAffineRegisterBlocking : Pass<"toyai-affine-register-blocking", "func::FuncOp"> {
  let summary = "Unroll-and-jam the loop surrounding each vectorized inner loop";
  let description = [{
    Unroll-and-jams the parent of every innermost vectorized loop by the
    largest divisor of its trip count that is <= `unroll`, then unrolls the
    innermost loop by `inner-unroll`, then forwards vector stores to later
    loads of the same address and deletes stores that are overwritten before
    being read. This keeps accumulators in registers across several
    reduction iterations. Relies on ToyAI buffers never aliasing.
  }];
  let options = [
    Option<"unrollJamFactor", "unroll", "unsigned", /*default=*/"4",
           "Maximum unroll-and-jam factor">,
    Option<"innerUnrollFactor", "inner-unroll", "unsigned", /*default=*/"2",
           "Unroll factor for the innermost vector loop">,
  ];
}

def ToyAIGPUTiling : Pass<"toyai-gpu-tiling", "func::FuncOp"> {
  let summary = "Tile scf.parallel loops into GPU blocks and threads";
  let description = [{
    First keeps reduction accumulators in registers: an scf.for that loads
    and stores the same loop-invariant element (C[i, j] in matmul) becomes
    an scf.for with an iter_arg, so each thread reads and writes its output
    once instead of once per reduction step.

    Then tiles the innermost two dimensions of every top-level scf.parallel
    loop by (block-y, block-x) and annotates the outer loop with block
    mapping and the intra-tile loop with thread mapping so that
    `convert-parallel-loops-to-gpu` produces a 2-D grid of 2-D blocks.
    Remaining outer dimensions are mapped to grid z / sequential loops.
  }];
  let options = [
    Option<"blockX", "block-x", "int64_t", /*default=*/"16", "Threads per block along x">,
    Option<"blockY", "block-y", "int64_t", /*default=*/"16", "Threads per block along y">,
  ];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::affine::AffineDialect",
  ];
}

def ToyAIGPUDataMovement : Pass<"toyai-gpu-data-movement", "func::FuncOp"> {
  let summary = "Place buffers on the device and insert host<->device copies";
  let description = [{
    Allocates a device buffer (`gpu.alloc`) for every memref argument and
    copies inputs to the device on entry and outputs back on exit
    (`gpu.memcpy`). Function-local `memref.alloc`s become device allocations.
    All uses inside the function are redirected to the device buffers, so
    every computation must run in GPU kernels.
  }];
  let dependentDialects = ["::mlir::gpu::GPUDialect"];
}

#endif // TOYAI_PASSES_TD
