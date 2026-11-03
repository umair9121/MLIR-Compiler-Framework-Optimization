#ifndef TOYAI_OPS_TD
#define TOYAI_OPS_TD

include "ToyAI/ToyAIDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Structured ops
//===----------------------------------------------------------------------===//

def ToyAI_MatmulOp : ToyAI_Op<"matmul", [Pure, InferTypeOpAdaptor]> {
  let summary = "2-D matrix multiplication";
  let description = [{
    Computes `result[i, j] = sum_k lhs[i, k] * rhs[k, j]`.

    ```mlir
    %c = toyai.matmul %a, %b : (tensor<64x32xf32>, tensor<32x16xf32>) -> tensor<64x16xf32>
    ```
  }];
  let arguments = (ins ToyAI_Tensor:$lhs, ToyAI_Tensor:$rhs);
  let results = (outs ToyAI_Tensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def ToyAI_Conv2DOp : ToyAI_Op<"conv2d", [Pure, InferTypeOpAdaptor]> {
  let summary = "2-D convolution (NCHW input, FCHW filter)";
  let description = [{
    Cross-correlates an `N x C x H x W` input with an `F x C x KH x KW`
    filter, producing `N x F x OH x OW` where
    `OH = (H + 2 * padding[0] - KH) / strides[0] + 1` (likewise for `OW`).
    Padding is symmetric and filled with zeros.

    ```mlir
    %y = toyai.conv2d %x, %w {padding = array<i64: 1, 1>, strides = array<i64: 1, 1>}
        : (tensor<1x3x32x32xf32>, tensor<8x3x3x3xf32>) -> tensor<1x8x32x32xf32>
    ```
  }];
  let arguments = (ins
    ToyAI_Tensor:$input,
    ToyAI_Tensor:$filter,
    DefaultValuedAttr<DenseI64ArrayAttr, "{1, 1}">:$strides,
    DefaultValuedAttr<DenseI64ArrayAttr, "{0, 0}">:$padding
  );
  let results = (outs ToyAI_Tensor:$result);
  let assemblyFormat = "$input `,` $filter attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Elementwise ops
//===----------------------------------------------------------------------===//

def ToyAI_AddOp : ToyAI_Op<"add", [Pure, Commutative, SameOperandsAndResultType]> {
  let summary = "elementwise addition";
  let arguments = (ins ToyAI_Tensor:$lhs, ToyAI_Tensor:$rhs);
  let results = (outs ToyAI_Tensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def ToyAI_ReluOp : ToyAI_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "elementwise rectified linear unit, max(x, 0)";
  let arguments = (ins ToyAI_Tensor:$input);
  let results = (outs ToyAI_Tensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

def ToyAI_ConstantOp : ToyAI_Op<"constant", [Pure]> {
  let summary = "tensor filled with a single scalar value";
  let description = [{
    ```mlir
    %z = toyai.constant 0.000000e+00 : tensor<4x4xf32>
    ```
  }];
  let arguments = (ins F32Attr:$value);
  let results = (outs ToyAI_Tensor:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
}

#endif // TOYAI_OPS_TD
