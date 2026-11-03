// RUN: toyai-opt %s -toyai-affine-cache-tiling -canonicalize -split-input-file | FileCheck %s
// RUN: toyai-opt %s -toyai-affine-cache-tiling="l1-kb=8 l2-kb=64" -canonicalize -split-input-file | FileCheck %s --check-prefix=SMALL

// i-j-k matmul is interchanged to i-k-j and tiled (k for L1, j for rows).
// CHECK-LABEL: func.func @matmul
// CHECK:      affine.for %[[KK:.*]] = 0 to 512 step 32 {
// CHECK-NEXT:   affine.for %[[JJ:.*]] = 0 to 256 step 128 {
// CHECK-NEXT:     affine.for %[[I:.*]] = 0 to 256 {
// CHECK-NEXT:       affine.for %[[K:.*]] = #{{.*}}(%[[KK]]) to #{{.*}}(%[[KK]]) {
// CHECK-NEXT:         affine.for %[[J:.*]] = #{{.*}}(%[[JJ]]) to #{{.*}}(%[[JJ]]) {
// CHECK-NEXT:           affine.load %{{.*}}[%[[I]], %[[K]]]
// CHECK-NEXT:           affine.load %{{.*}}[%[[K]], %[[J]]]
// CHECK-NEXT:           affine.load %{{.*}}[%[[I]], %[[J]]]
// CHECK:      } {toyai.tile_sizes = array<i64: 256, 32, 128>}

// With small caches the i loop gets tiled too.
// SMALL-LABEL: func.func @matmul
// SMALL: toyai.tile_sizes = array<i64: 32, 8, 128>
func.func @matmul(%A: memref<256x512xf32>, %B: memref<512x256xf32>, %C: memref<256x256xf32>) {
  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 256 {
      affine.for %k = 0 to 512 {
        %a = affine.load %A[%i, %k] : memref<256x512xf32>
        %b = affine.load %B[%k, %j] : memref<512x256xf32>
        %c = affine.load %C[%i, %j] : memref<256x256xf32>
        %p = arith.mulf %a, %b : f32
        %s = arith.addf %c, %p : f32
        affine.store %s, %C[%i, %j] : memref<256x256xf32>
      }
    }
  }
  return
}

// -----

// Small loops are not tiled; the nest is only annotated.
// CHECK-LABEL: func.func @small
// CHECK:      affine.for %{{.*}} = 0 to 8 {
// CHECK-NEXT:   affine.for %{{.*}} = 0 to 8 {
// CHECK:      } {toyai.tile_sizes = array<i64: 8, 8>}
func.func @small(%A: memref<8x8xf32>) {
  %cst = arith.constant 1.0 : f32
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      affine.store %cst, %A[%i, %j] : memref<8x8xf32>
    }
  }
  return
}

// -----

// A column-major walk is interchanged so the inner loop is unit stride.
// CHECK-LABEL: func.func @transpose_walk
// CHECK:      affine.for %[[I:.*]] = 0 to 64 {
// CHECK-NEXT:   affine.for %[[J:.*]] = 0 to 64 {
// CHECK-NEXT:     affine.load %{{.*}}[%[[I]], %[[J]]]
func.func @transpose_walk(%A: memref<64x64xf32>, %B: memref<64x64xf32>) {
  affine.for %j = 0 to 64 {
    affine.for %i = 0 to 64 {
      %v = affine.load %A[%i, %j] : memref<64x64xf32>
      affine.store %v, %B[%i, %j] : memref<64x64xf32>
    }
  }
  return
}
