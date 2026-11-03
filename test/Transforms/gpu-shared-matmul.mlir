// RUN: toyai-opt %s -toyai-gpu-shared-matmul -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @matmul
// One 16x16-thread block per 64x64 tile of C: a 2x1 grid for 64x128.
// CHECK:       gpu.launch blocks({{[^)]*}}) in (%{{[a-z0-9]*}} = %c2, %{{[a-z0-9]*}} = %c1{{[_0-9]*}}, %{{[a-z0-9]*}} = %c1{{[_0-9]*}})
// CHECK-SAME:    threads({{[^)]*}}) in (%{{[a-z0-9]*}} = %c16{{[_0-9]*}}, %{{[a-z0-9]*}} = %c16{{[_0-9]*}}, %{{[a-z0-9]*}} = %c1{{[_0-9]*}})
// CHECK-SAME:    memref<64x16xf32, #gpu.address_space<workgroup>>
// CHECK-SAME:    memref<16x64xf32, #gpu.address_space<workgroup>>
// CHECK-COUNT-16: memref.load %arg2[{{.*}}] : memref<64x128xf32>
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args({{.*}}) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
// CHECK-COUNT-4:   memref.store {{.*}} : memref<64x16xf32, #gpu.address_space<workgroup>>
// CHECK:           gpu.barrier
// CHECK:           scf.for
// CHECK-COUNT-4:     memref.load {{.*}} : memref<64x16xf32, #gpu.address_space<workgroup>>
// CHECK-COUNT-16:    arith.mulf
// CHECK:           gpu.barrier
// CHECK-COUNT-16: memref.store %{{.*}}, %arg2[{{.*}}] : memref<64x128xf32>
// CHECK:         gpu.terminator
// CHECK-NOT:   scf.parallel
func.func @matmul(%A: memref<64x32xf32>, %B: memref<32x128xf32>, %C: memref<64x128xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  scf.parallel (%i, %j) = (%c0, %c0) to (%c64, %c128) step (%c1, %c1) {
    scf.for %k = %c0 to %c32 step %c1 {
      %a = memref.load %A[%i, %k] : memref<64x32xf32>
      %b = memref.load %B[%k, %j] : memref<32x128xf32>
      %c = memref.load %C[%i, %j] : memref<64x128xf32>
      %p = arith.mulf %a, %b : f32
      %s = arith.addf %c, %p : f32
      memref.store %s, %C[%i, %j] : memref<64x128xf32>
    }
    scf.reduce
  }
  return
}

// -----

// Sizes that do not divide into 64x64 tiles keep the generic path.
// CHECK-LABEL: func.func @odd
// CHECK-NOT:   gpu.launch
// CHECK:       scf.parallel
func.func @odd(%A: memref<30x32xf32>, %B: memref<32x128xf32>, %C: memref<30x128xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c30 = arith.constant 30 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  scf.parallel (%i, %j) = (%c0, %c0) to (%c30, %c128) step (%c1, %c1) {
    scf.for %k = %c0 to %c32 step %c1 {
      %a = memref.load %A[%i, %k] : memref<30x32xf32>
      %b = memref.load %B[%k, %j] : memref<32x128xf32>
      %c = memref.load %C[%i, %j] : memref<30x128xf32>
      %p = arith.mulf %a, %b : f32
      %s = arith.addf %c, %p : f32
      memref.store %s, %C[%i, %j] : memref<30x128xf32>
    }
    scf.reduce
  }
  return
}
