add_mlir_dialect_library(MLIRToyAI
  IR/ToyAIDialect.cpp
  IR/ToyAIOps.cpp
  Conversion/ToyAIToLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ToyAI

  DEPENDS
  MLIRToyAIOpsIncGen
  MLIRToyAIPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRLinalgDialect
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRTensorDialect
  MLIRTransforms
  )