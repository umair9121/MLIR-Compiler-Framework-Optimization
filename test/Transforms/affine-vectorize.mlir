// RUN: toyai-opt %s -toyai-affine-vectorize -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @axpy_row
// CHECK:       affine.for %[[J:.*]] = 0 to 64 step 8 {
// CHECK:         %[[A:.*]] = affine.load %{{.*}}[%{{.*}}] : memref<64xf32>
// CHECK:         %[[AV:.*]] = vector.broadcast %[[A]] : f32 to vector<8xf32>
// CHECK:         %[[B:.*]] = affine.vector_load %{{.*}}[%{{.*}}, %[[J]]] : memref<64x64xf32>, vector<8xf32>
// CHECK:         %[[C:.*]] = affine.vector_load %{{.*}}[%{{.*}}, %[[J]]] : memref<64x64xf32>, vector<8xf32>
// CHECK:         %[[M:.*]] = arith.mulf %[[AV]], %[[B]] : vector<8xf32>
// CHECK:         %[[S:.*]] = arith.addf %[[C]], %[[M]] : vector<8xf32>
// CHECK:         affine.vector_store %[[S]], %{{.*}}[%{{.*}}, %[[J]]] : memref<64x64xf32>, vector<8xf32>
func.func @axpy_row(%A: memref<64xf32>, %B: memref<64x64xf32>, %C: memref<64x64xf32>, %k: index, %i: index) {
  affine.for %j = 0 to 64 {
    %a = affine.load %A[%k] : memref<64xf32>
    %b = affine.load %B[%k, %j] : memref<64x64xf32>
    %c = affine.load %C[%i, %j] : memref<64x64xf32>
    %p = arith.mulf %a, %b : f32
    %s = arith.addf %c, %p : f32
    affine.store %s, %C[%i, %j] : memref<64x64xf32>
  }
  return
}

// -----

// Scalars from outside the loop and constants become splats.
// CHECK-LABEL: func.func @relu_fill
// CHECK:       %[[F:.*]] = vector.broadcast %{{.*}} : f32 to vector<8xf32>
// CHECK:       affine.for %{{.*}} = 0 to 32 step 8 {
// CHECK:         %[[Z:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
// CHECK:         arith.maximumf %{{.*}}, %[[Z]] : vector<8xf32>
// CHECK:         affine.vector_store %[[F]]
func.func @relu_fill(%A: memref<32xf32>, %B: memref<32xf32>, %fill: f32) {
  affine.for %j = 0 to 32 {
    %zero = arith.constant 0.0 : f32
    %a = affine.load %A[%j] : memref<32xf32>
    %r = arith.maximumf %a, %zero : f32
    affine.store %r, %B[%j] : memref<32xf32>
    affine.store %fill, %A[%j] : memref<32xf32>
  }
  return
}

// -----

// Not vectorized: strided access, trip count not a multiple of 8, and a
// reduction into an invariant location.
// CHECK-LABEL: func.func @rejected
// CHECK-NOT:   vector
func.func @rejected(%A: memref<64x64xf32>, %B: memref<12xf32>, %s: memref<1xf32>) {
  affine.for %j = 0 to 64 {
    %a = affine.load %A[%j, 0] : memref<64x64xf32>
    affine.store %a, %A[0, %j] : memref<64x64xf32>
  }
  affine.for %j = 0 to 12 {
    %b = affine.load %B[%j] : memref<12xf32>
    affine.store %b, %B[%j] : memref<12xf32>
  }
  affine.for %j = 0 to 64 {
    %a = affine.load %A[0, %j] : memref<64x64xf32>
    %acc = affine.load %s[0] : memref<1xf32>
    %sum = arith.addf %acc, %a : f32
    affine.store %sum, %s[0] : memref<1xf32>
  }
  return
}
