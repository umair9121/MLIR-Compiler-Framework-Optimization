add_mlir_dialect(ToyAIOps toyai)
add_mlir_doc(ToyAIOps ToyAIOps ToyAI/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ToyAI)
add_public_tablegen_target(MLIRToyAIPassIncGen)
add_mlir_doc(Passes ToyAIPasses ToyAI/ -gen-pass-doc)
