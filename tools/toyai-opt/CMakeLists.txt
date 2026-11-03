get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  MLIRToLLVMIRTranslationRegistration
  MLIRToyAI
  )

add_llvm_executable(toyai-opt toyai-opt.cpp)
llvm_update_compile_flags(toyai-opt)
target_link_libraries(toyai-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(toyai-opt)
