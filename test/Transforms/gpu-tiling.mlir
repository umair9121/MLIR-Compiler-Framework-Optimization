// RUN: toyai-opt %s -toyai-gpu-tiling="block-x=32 block-y=8" -split-input-file | FileCheck %s

// The 2-D output space is split into 8x32 blocks; the last (unit-stride)
// dimension maps to x. The k reduction keeps C[i, j] in a register.
// CHECK-LABEL: func.func @matmul
// CHECK:       %[[SY:.*]] = arith.muli %{{.*}}, %c8 : index
// CHECK:       %[[SX:.*]] = arith.muli %{{.*}}, %c32{{.*}} : index
// CHECK:       scf.parallel (%[[BI:.*]], %[[BJ:.*]]) = ({{.*}}) to ({{.*}}) step (%[[SY]], %[[SX]])
// CHECK:         scf.parallel (%[[TI:.*]], %[[TJ:.*]]) = ({{.*}}) to (%[[SY]], %[[SX]])
// CHECK:           %[[INIT:.*]] = memref.load %[[C:.*]][%{{.*}}, %{{.*}}]
// CHECK:           %[[SUM:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]]) -> (f32) {
// CHECK-NOT:         memref.store
// CHECK:             scf.yield
// CHECK:           memref.store %[[SUM]], %[[C]]
// CHECK:         {mapping = [#gpu.loop_dim_map<processor = thread_y, map = (d0) -> (d0), bound = (d0) -> (d0)>, #gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// CHECK:       {mapping = [#gpu.loop_dim_map<processor = block_y, map = (d0) -> (d0), bound = (d0) -> (d0)>, #gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
func.func @matmul(%A: memref<64x32xf32>, %B: memref<32x96xf32>, %C: memref<64x96xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c96 = arith.constant 96 : index
  scf.parallel (%i, %j) = (%c0, %c0) to (%c64, %c96) step (%c1, %c1) {
    scf.for %k = %c0 to %c32 step %c1 {
      %a = memref.load %A[%i, %k] : memref<64x32xf32>
      %b = memref.load %B[%k, %j] : memref<32x96xf32>
      %c = memref.load %C[%i, %j] : memref<64x96xf32>
      %p = arith.mulf %a, %b : f32
      %s = arith.addf %c, %p : f32
      memref.store %s, %C[%i, %j] : memref<64x96xf32>
    }
  }
  return
}

// -----

// 4-D loops: (n, f) go to block z / sequential, (h, w) to a 2-D grid.
// CHECK-LABEL: func.func @nchw
// CHECK: processor = sequential{{.*}}processor = block_z{{.*}}processor = block_y{{.*}}processor = block_x
func.func @nchw(%A: memref<2x4x16x64xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %zero = arith.constant 0.0 : f32
  scf.parallel (%n, %f, %h, %w) = (%c0, %c0, %c0, %c0) to (%c2, %c4, %c16, %c64) step (%c1, %c1, %c1, %c1) {
    memref.store %zero, %A[%n, %f, %h, %w] : memref<2x4x16x64xf32>
  }
  return
}
