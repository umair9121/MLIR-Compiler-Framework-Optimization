# Entry point so CMake's add_lit_testsuite can drive the pip-installed `lit`
# (LLVM installs don't ship llvm-lit).
from lit.main import main

if __name__ == "__main__":
    main()
