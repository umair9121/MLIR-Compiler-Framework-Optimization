add_mlir_dialect_library(MLIRToyAI
  IR/ToyAIDialect.cpp
  IR/ToyAIOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ToyAI

  DEPENDS
  MLIRToyAIOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  )