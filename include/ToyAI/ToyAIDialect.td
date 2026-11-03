#ifndef TOYAI_DIALECT_TD
#define TOYAI_DIALECT_TD

include "mlir/IR/OpBase.td"

def ToyAI_Dialect : Dialect {
  let name = "toyai";
  let summary = "A tiny tensor dialect for machine-learning kernels";
  let description = [{
    ToyAI models a handful of high-level ML operations (matmul, conv2d,
    elementwise add/relu, constants) on statically shaped f32 tensors. It is
    lowered to `linalg` on tensors and from there to tiled `affine` loops for
    CPUs or to `gpu` kernels for NVIDIA GPUs.
  }];
  let cppNamespace = "::mlir::toyai";
}

class ToyAI_Op<string mnemonic, list<Trait> traits = []>
    : Op<ToyAI_Dialect, mnemonic, traits>;

// All ToyAI values are statically shaped f32 tensors.
def ToyAI_Tensor : Type<
    And<[TensorRankOf<[F32], [1, 2, 3, 4]>.predicate, HasStaticShapePred]>,
    "statically shaped f32 tensor of rank 1 to 4", "::mlir::RankedTensorType">;

#endif // TOYAI_DIALECT_TD
