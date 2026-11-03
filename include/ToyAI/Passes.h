#ifndef TOYAI_PASSES_H
#define TOYAI_PASSES_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace toyai {

#define GEN_PASS_DECL
#include "ToyAI/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "ToyAI/Passes.h.inc"

} // namespace toyai
} // namespace mlir

#endif // TOYAI_PASSES_H
