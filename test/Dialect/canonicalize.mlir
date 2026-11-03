// RUN: toyai-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: func.func @relu_idempotent
// CHECK: %[[R:.*]] = toyai.relu %arg0
// CHECK-NOT: toyai.relu
// CHECK: return %[[R]]
func.func @relu_idempotent(%a: tensor<8xf32>) -> tensor<8xf32> {
  %0 = toyai.relu %a : (tensor<8xf32>) -> tensor<8xf32>
  %1 = toyai.relu %0 : (tensor<8xf32>) -> tensor<8xf32>
  %2 = toyai.relu %1 : (tensor<8xf32>) -> tensor<8xf32>
  return %2 : tensor<8xf32>
}
