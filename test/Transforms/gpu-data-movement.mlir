// RUN: toyai-opt %s -toyai-gpu-data-movement | FileCheck %s

// Inputs are uploaded, the output (written before it is read) is only
// downloaded, temporaries live on the device, everything is freed.
// CHECK-LABEL: func.func @f
// CHECK-SAME:  (%[[A:.*]]: memref<16xf32>, %[[OUT:.*]]: memref<16xf32>)
// CHECK:       %[[DA:.*]] = gpu.alloc () : memref<16xf32>
// CHECK-NEXT:  gpu.memcpy %[[DA]], %[[A]]
// CHECK-NEXT:  %[[DOUT:.*]] = gpu.alloc () : memref<16xf32>
// CHECK-NOT:   gpu.memcpy %[[DOUT]], %[[OUT]]
// CHECK:       %[[TMP:.*]] = gpu.alloc () : memref<16xf32>
// CHECK:       memref.load %[[DA]]
// CHECK:       memref.store %{{.*}}, %[[TMP]]
// CHECK:       memref.store %{{.*}}, %[[DOUT]]
// CHECK-NOT:   memref.dealloc
// CHECK:       gpu.memcpy %[[OUT]], %[[DOUT]]
// CHECK-DAG:   gpu.dealloc %[[TMP]]
// CHECK-DAG:   gpu.dealloc %[[DA]]
// CHECK-DAG:   gpu.dealloc %[[DOUT]]
// CHECK:       return
func.func @f(%A: memref<16xf32>, %Out: memref<16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %tmp = memref.alloc() : memref<16xf32>
  scf.parallel (%i) = (%c0) to (%c16) step (%c1) {
    %a = memref.load %A[%i] : memref<16xf32>
    %d = arith.addf %a, %a : f32
    memref.store %d, %tmp[%i] : memref<16xf32>
  }
  scf.parallel (%i) = (%c0) to (%c16) step (%c1) {
    %t = memref.load %tmp[%i] : memref<16xf32>
    memref.store %t, %Out[%i] : memref<16xf32>
  }
  memref.dealloc %tmp : memref<16xf32>
  return
}
