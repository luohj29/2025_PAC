#include <iostream>
#include "./src/kernels.h"
#include <map>
#include <tuple>
#include <functional>
#include <cstring>
#include <cstdlib>  // for aligned_alloc / free

using namespace std;

#define compute
#ifndef compute

int main(int argc){

}
#else

int main(int argc, char *argv[])
{
    int number = atoi(argv[1]);
    int size = atoi(argv[2]);
    map<int, function<void(double *, double *, double *, int, int, int)>> cpu_gemm_versions = {
        {0, cpu_gemm_blas<double>},
        {1, cpu_gemm_v1<double,64,64>},
        {2, cpu_gemm_v2<double, 64, 64>},
    };

    double* A = (double*)aligned_alloc(32, sizeof(double) * size * size);
    double* B = (double*)aligned_alloc(32, sizeof(double) * size * size);
    double* C = (double*)aligned_alloc(32, sizeof(double) * size * size);

    auto gemm_func = cpu_gemm_versions[number];
    memset(C, 0, size * size * sizeof(double));
    
    gemm_func(A, B, C, size, size, size);


    free(A);
    free(B);
    free(C);
        

}
#endif