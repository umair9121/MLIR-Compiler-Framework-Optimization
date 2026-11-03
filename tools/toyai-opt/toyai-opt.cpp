//===- toyai-opt.cpp - ToyAI optimizer driver -----------------------------===//

#include "ToyAI/Passes.h"
#include "ToyAI/Pipelines.h"
#include "ToyAI/ToyAIDialect.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::toyai::registerToyAIPasses();
  mlir::toyai::registerToyAIPipelines();

  mlir::DialectRegistry registry;
  registry.insert<mlir::toyai::ToyAIDialect>();
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllGPUToLLVMIRTranslations(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "ToyAI optimizer driver\n", registry));
}
