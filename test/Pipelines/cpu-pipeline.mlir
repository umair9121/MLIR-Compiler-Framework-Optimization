// RUN: toyai-opt %s --pass-pipeline="builtin.module(toyai-cpu-pipeline)" | FileCheck %s
// RUN: toyai-opt %s --pass-pipeline="builtin.module(toyai-cpu-pipeline{optimize=false})" | FileCheck %s --check-prefix=NAIVE

// The whole program ends up in the LLVM dialect, with the C entry point
// taking the result as a trailing out-parameter.
// CHECK-LABEL: llvm.func @mlp
// CHECK:       llvm.fmul {{.*}} : vector<8xf32>
// CHECK:       llvm.func @_mlir_ciface_mlp(%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr)
// CHECK-NOT:   toyai.
// CHECK-NOT:   affine.
// CHECK-NOT:   linalg.

// NAIVE-LABEL: llvm.func @mlp
// NAIVE-NOT:   vector<8xf32>
// NAIVE:       llvm.func @_mlir_ciface_mlp
func.func @mlp(%x: tensor<64x256xf32>, %w: tensor<256x128xf32>, %b: tensor<64x128xf32>) -> tensor<64x128xf32>
    attributes {llvm.emit_c_interface} {
  %0 = toyai.matmul %x, %w : (tensor<64x256xf32>, tensor<256x128xf32>) -> tensor<64x128xf32>
  %1 = toyai.add %0, %b : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  %2 = toyai.relu %1 : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %2 : tensor<64x128xf32>
}
