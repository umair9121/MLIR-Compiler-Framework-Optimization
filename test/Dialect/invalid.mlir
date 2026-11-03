// RUN: toyai-opt %s -split-input-file -verify-diagnostics

func.func @matmul_mismatch(%a: tensor<4x8xf32>, %b: tensor<4x2xf32>) -> tensor<4x2xf32> {
  // expected-error @+2 {{failed to infer returned types}}
  // expected-error @+1 {{contraction dimension mismatch}}
  %0 = toyai.matmul %a, %b : (tensor<4x8xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// -----

func.func @matmul_wrong_result(%a: tensor<4x8xf32>, %b: tensor<8x2xf32>) -> tensor<4x3xf32> {
  // expected-error @+2 {{failed to infer returned types}}
  // expected-error @+1 {{inferred type(s) 'tensor<4x2xf32>' are incompatible with return type(s) of operation 'tensor<4x3xf32>'}}
  %0 = toyai.matmul %a, %b : (tensor<4x8xf32>, tensor<8x2xf32>) -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}

// -----

func.func @matmul_rank(%a: tensor<4x8x2xf32>, %b: tensor<8x2xf32>) -> tensor<4x2xf32> {
  // expected-error @+2 {{failed to infer returned types}}
  // expected-error @+1 {{expects rank-2 operands}}
  %0 = toyai.matmul %a, %b : (tensor<4x8x2xf32>, tensor<8x2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// -----

func.func @conv_channels(%x: tensor<1x3x8x8xf32>, %k: tensor<4x2x3x3xf32>) -> tensor<1x4x6x6xf32> {
  // expected-error @+2 {{failed to infer returned types}}
  // expected-error @+1 {{channel mismatch: input has 3 channels, filter expects 2}}
  %0 = toyai.conv2d %x, %k : (tensor<1x3x8x8xf32>, tensor<4x2x3x3xf32>) -> tensor<1x4x6x6xf32>
  return %0 : tensor<1x4x6x6xf32>
}

// -----

func.func @conv_too_big(%x: tensor<1x1x2x2xf32>, %k: tensor<1x1x3x3xf32>) -> tensor<1x1x1x1xf32> {
  // expected-error @+2 {{failed to infer returned types}}
  // expected-error @+1 {{does not fit the (padded) input}}
  %0 = toyai.conv2d %x, %k : (tensor<1x1x2x2xf32>, tensor<1x1x3x3xf32>) -> tensor<1x1x1x1xf32>
  return %0 : tensor<1x1x1x1xf32>
}

// -----

func.func @add_mismatch(%a: tensor<4x2xf32>, %b: tensor<2x4xf32>) -> tensor<4x2xf32> {
  // expected-error @+1 {{requires the same type for all operands and results}}
  %0 = toyai.add %a, %b : (tensor<4x2xf32>, tensor<2x4xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// -----

func.func @dynamic(%a: tensor<?x2xf32>) -> tensor<?x2xf32> {
  // expected-error @+1 {{must be statically shaped f32 tensor of rank 1 to 4}}
  %0 = toyai.relu %a : (tensor<?x2xf32>) -> tensor<?x2xf32>
  return %0 : tensor<?x2xf32>
}
