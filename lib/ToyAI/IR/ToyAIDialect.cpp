#include "ToyAI/ToyAIDialect.h"
#include "ToyAI/ToyAIOps.h"

using namespace mlir;
using namespace mlir::toyai;

#include "ToyAI/ToyAIOpsDialect.cpp.inc"

void ToyAIDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ToyAI/ToyAIOps.cpp.inc"
      >();
}
