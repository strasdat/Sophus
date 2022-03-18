find . -type d \( -path ./sympy -o -path ./doxyrest_b -o -path "./*/CMakeFiles/*" \) -prune -o \( -iname "*.hpp" -o -iname "*.cpp" \) -print | xargs clang-format -i
