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

#endif // TOYAI_PASSES_TD
