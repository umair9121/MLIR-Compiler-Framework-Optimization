// RUN: toyai-opt %s | toyai-opt | FileCheck %s

// CHECK-LABEL: func.func @mlp
func.func @mlp(%x: tensor<4x8xf32>, %w: tensor<8x2xf32>, %b: tensor<4x2xf32>) -> tensor<4x2xf32> {
  // CHECK: %[[MM:.*]] = toyai.matmul %{{.*}}, %{{.*}} : (tensor<4x8xf32>, tensor<8x2xf32>) -> tensor<4x2xf32>
  %0 = toyai.matmul %x, %w : (tensor<4x8xf32>, tensor<8x2xf32>) -> tensor<4x2xf32>
  // CHECK: %[[ADD:.*]] = toyai.add %[[MM]], %{{.*}} : (tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
  %1 = toyai.add %0, %b : (tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
  // CHECK: toyai.relu %[[ADD]] : (tensor<4x2xf32>) -> tensor<4x2xf32>
  %2 = toyai.relu %1 : (tensor<4x2xf32>) -> tensor<4x2xf32>
  return %2 : tensor<4x2xf32>
}

// CHECK-LABEL: func.func @conv
func.func @conv(%x: tensor<1x3x8x8xf32>, %k: tensor<4x3x3x3xf32>) -> tensor<1x4x4x4xf32> {
  // CHECK: toyai.conv2d %{{.*}}, %{{.*}} {padding = array<i64: 1, 1>, strides = array<i64: 2, 2>}
  %0 = toyai.conv2d %x, %k {padding = array<i64: 1, 1>, strides = array<i64: 2, 2>}
      : (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>) -> tensor<1x4x4x4xf32>
  return %0 : tensor<1x4x4x4xf32>
}

// CHECK-LABEL: func.func @constant
func.func @constant() -> tensor<2x3xf32> {
  // CHECK: toyai.constant 2.500000e+00 : tensor<2x3xf32>
  %0 = toyai.constant 2.5 : tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
