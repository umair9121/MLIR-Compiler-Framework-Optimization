add_mlir_dialect_library(MLIRToyAI
  IR/ToyAIDialect.cpp
  IR/ToyAIOps.cpp
  Conversion/ToyAIToLinalg.cpp
  Transforms/AffineCacheTiling.cpp
  Transforms/AffineRegisterBlocking.cpp
  Transforms/AffineVectorize.cpp
  Transforms/GPUDataMovement.cpp
  Transforms/GPUTiling.cpp
  Transforms/MemrefCopyToLinalg.cpp
  Pipelines/Pipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ToyAI

  DEPENDS
  MLIRToyAIOpsIncGen
  MLIRToyAIPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineAnalysis
  MLIRAffineDialect
  MLIRAffineTransforms
  MLIRAffineUtils
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRGPUPipelines
  MLIRGPUTransforms
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSideEffectInterfaces
  MLIRTensorDialect
  MLIRTransforms
  MLIRVectorDialect
  )
