#ifndef TOYAI_TOYAIOPS_H
#define TOYAI_TOYAIOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "ToyAI/ToyAIDialect.h"

#define GET_OP_CLASSES
#include "ToyAI/ToyAIOps.h.inc"

#endif // TOYAI_TOYAIOPS_H
