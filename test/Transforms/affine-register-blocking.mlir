// RUN: toyai-opt %s -toyai-affine-register-blocking="unroll=4 inner-unroll=1" | FileCheck %s

// Unroll-and-jam k by 4; C is loaded once and stored once per 4 k steps.
// CHECK-LABEL: func.func @kj
// CHECK:       affine.for %[[K:.*]] = 0 to 32 step 4 {
// CHECK:         affine.for %[[J:.*]] = 0 to 64 step 8 {
// CHECK-COUNT-1:   affine.vector_load %[[C:.*]][%{{.*}}, %[[J]]] : memref<64x64xf32>
// CHECK-COUNT-4:   affine.vector_load %{{.*}}[%{{.*}}, %[[J]]] : memref<32x64xf32>
// CHECK-NOT:       affine.vector_load
// CHECK-COUNT-1:   affine.vector_store
// CHECK-NOT:       affine.vector_store
// CHECK:         }
func.func @kj(%a: vector<8xf32>, %B: memref<32x64xf32>, %C: memref<64x64xf32>, %i: index) {
  affine.for %k = 0 to 32 {
    affine.for %j = 0 to 64 step 8 {
      %c = affine.vector_load %C[%i, %j] : memref<64x64xf32>, vector<8xf32>
      %b = affine.vector_load %B[%k, %j] : memref<32x64xf32>, vector<8xf32>
      %p = arith.mulf %a, %b : vector<8xf32>
      %s = arith.addf %c, %p : vector<8xf32>
      affine.vector_store %s, %C[%i, %j] : memref<64x64xf32>, vector<8xf32>
    }
  }
  return
}
