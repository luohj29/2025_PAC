find_package(OpenMP REQUIRED)

aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)
add_executable(cpu_gemm main.cpp  ${SRC})
# add_executable(test_gemm test_gemm.cpp ${SRC})
target_compile_options(cpu_gemm PRIVATE -fopenmp -O3 -march=native -lopenblas)
target_link_libraries(cpu_gemm PRIVATE OpenMP::OpenMP_CXX)
find_package(OpenBLAS REQUIRED)
target_link_libraries(cpu_gemm PRIVATE /lib/x86_64-linux-gnu/libopenblas.so)