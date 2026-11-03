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

#endif // TOYAI_PASSES_TD
