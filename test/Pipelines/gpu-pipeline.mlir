// RUN: toyai-opt %s --pass-pipeline="builtin.module(toyai-gpu-pipeline{format=isa})" | FileCheck %s

// Kernels are compiled to PTX for sm_86 and embedded in a gpu.binary; the
// host side is plain LLVM dialect calling the MLIR CUDA runtime wrappers.
// CHECK:     llvm.func @mlp
// CHECK:       llvm.call @mgpuStreamCreate
// CHECK:       llvm.call @mgpuMemAlloc
// CHECK:       llvm.call @mgpuMemcpy
// CHECK:       gpu.launch_func <%{{.*}} : !llvm.ptr> @mlp_kernel::@mlp_kernel
// CHECK:       llvm.call @mgpuStreamSynchronize
// CHECK:     llvm.func @_mlir_ciface_mlp
// CHECK:     gpu.binary @mlp_kernel
// CHECK-SAME: #gpu.object<#nvvm.target<O = 3, chip = "sm_86">
// CHECK-SAME: .target sm_86
func.func @mlp(%x: tensor<64x256xf32>, %w: tensor<256x128xf32>, %b: tensor<64x128xf32>) -> tensor<64x128xf32>
    attributes {llvm.emit_c_interface} {
  %0 = toyai.matmul %x, %w : (tensor<64x256xf32>, tensor<256x128xf32>) -> tensor<64x128xf32>
  %1 = toyai.add %0, %b : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  %2 = toyai.relu %1 : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %2 : tensor<64x128xf32>
}
