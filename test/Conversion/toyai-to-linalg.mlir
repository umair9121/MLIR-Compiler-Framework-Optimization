// RUN: toyai-opt %s -convert-toyai-to-linalg | FileCheck %s

// CHECK-LABEL: func.func @matmul
// CHECK-SAME: (%[[A:.*]]: tensor<16x32xf32>, %[[B:.*]]: tensor<32x8xf32>)
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<16x8xf32>
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[EMPTY]] : tensor<16x8xf32>)
// CHECK: %[[MM:.*]] = linalg.matmul ins(%[[A]], %[[B]] : tensor<16x32xf32>, tensor<32x8xf32>) outs(%[[FILL]] : tensor<16x8xf32>)
// CHECK: return %[[MM]]
func.func @matmul(%a: tensor<16x32xf32>, %b: tensor<32x8xf32>) -> tensor<16x8xf32> {
  %0 = toyai.matmul %a, %b : (tensor<16x32xf32>, tensor<32x8xf32>) -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func.func @add_relu
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK: arith.addf
// CHECK: linalg.generic
// CHECK: arith.maximumf
// CHECK-NOT: toyai.
func.func @add_relu(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = toyai.add %a, %b : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = toyai.relu %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @conv_padded
// CHECK: %[[PAD:.*]] = tensor.pad %{{.*}} low[0, 0, 1, 1] high[0, 0, 1, 1]
// CHECK: tensor<1x3x8x8xf32> to tensor<1x3x10x10xf32>
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK-SAME: dilations = dense<1> : vector<2xi64>
// CHECK-SAME: strides = dense<2> : vector<2xi64>
// CHECK-SAME: ins(%[[PAD]], %{{.*}} : tensor<1x3x10x10xf32>, tensor<4x3x3x3xf32>)
func.func @conv_padded(%x: tensor<1x3x8x8xf32>, %k: tensor<4x3x3x3xf32>) -> tensor<1x4x4x4xf32> {
  %0 = toyai.conv2d %x, %k {padding = array<i64: 1, 1>, strides = array<i64: 2, 2>}
      : (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>) -> tensor<1x4x4x4xf32>
  return %0 : tensor<1x4x4x4xf32>
}

// CHECK-LABEL: func.func @conv_unpadded
// CHECK-NOT: tensor.pad
// CHECK: linalg.conv_2d_nchw_fchw
func.func @conv_unpadded(%x: tensor<1x3x8x8xf32>, %k: tensor<4x3x3x3xf32>) -> tensor<1x4x6x6xf32> {
  %0 = toyai.conv2d %x, %k : (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>) -> tensor<1x4x6x6xf32>
  return %0 : tensor<1x4x6x6xf32>
}

// CHECK-LABEL: func.func @constant
// CHECK: %[[C:.*]] = arith.constant 1.500000e+00 : f32
// CHECK: linalg.fill ins(%[[C]] : f32)
func.func @constant() -> tensor<2x2xf32> {
  %0 = toyai.constant 1.5 : tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
