#include <iostream>
#include <omp.h>
#include <map>
#include <tuple>
#include <cstring>
#include <functional>
#include <cmath>
#include <chrono>
#include "./src/kernels.h"  // 包含所有的 GEMM 实现
#include <cstdlib>  // for aligned_alloc / free

using namespace std;

// 比较两个矩阵是否近似相等
template <typename T>
bool matrix_equal(T *A, T *B, int size, double tol = 1e-8)
{
    for (int i = 0; i < size; ++i)
    {
        if (fabs(A[i] - B[i]) > tol)
            return false;
    }
    return true;
}

// 单元测试
void test_gemm(map<int, function<void(double *, double *, double *, int, int, int)>> &dict)
{
    const int M = 512, N = 512, K = 512;
    double *A, *B, *Cref,*Ctest;
    A = new double[M * K];
    B = new double[K * N];
    Cref = new double[M * N];
    Ctest = new double[M * N];
    for (int i = 0; i < M * K; ++i)
    {
        A[i] = static_cast<double>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < K * N; ++i)
    {
        B[i] = static_cast<double>(rand()) / RAND_MAX;
    }
    memset(Cref, 0, M * N * sizeof(double));
    memset(Ctest, 0, M * N * sizeof(double));

    cpu_gemm_blas(A, B, Cref, M, N, K); // 参考实现
    for (const auto &[version, gemm_func] : dict)
    {
        memset(Ctest, 0, M * N * sizeof(double));
        gemm_func(A, B, Ctest, M, N, K);
        if (matrix_equal(Cref, Ctest, M * N))
        {
            cout << "Version " << version << " passed the test." << endl;
        }
        else
        {
            cout << "Version " << version << " failed the test." << endl;
        }
    }
    cout << "Unit test completed." << endl;
    cout << "----------------------------------------" << endl;
    delete[] A;
    delete[] B;
    delete[] Cref;
    delete[] Ctest;

}

#define compute
#ifndef compute
int main(int argc, char *argv[])
{
    int number = atoi(argv[1]);
    int size = atoi(argv[2]);
    map<int, function<void(double *, double *, double *, int, int, int)>> cpu_gemm_versions = {
        {0, cpu_gemm_blas<double>},
        {1, cpu_gemm_v1<double,64,64>},
        {2, cpu_gemm_v3<double>},
    };

    double* A = (double*)aligned_alloc(32, sizeof(double) * size * size);
    double* B = (double*)aligned_alloc(32, sizeof(double) * size * size);
    double* C = (double*)aligned_alloc(32, sizeof(double) * size * size);

    auto gemm_func = cpu_gemm_versions[number];
    memset(C, 0, size * size * sizeof(double));
    auto start = chrono::high_resolution_clock::now();
    gemm_func(A, B, C, size, size, size);

    auto end = chrono::high_resolution_clock::now();
    double elapsed_time = chrono::duration<double>(end - start).count();
    cout << "Elapsed time: " << elapsed_time << " seconds" << endl;

    free(A);
    free(B);
    free(C);
        

}
#else
int main()
{
    map<int, tuple<int, int, int>> matrix_size = {
        {0, {512, 512, 512}},
        {1, {768, 768, 768}},
        {2, {1024, 1024, 1024}},
        {3, {2048, 2048, 2048}},
        // {6, {4096, 4096, 4096}},
        // {1, {12349, 140, 3040}},
        // {2, {24698, 8, 3040}},
        // {3, {256, 3040, 3040}},
        // {4, {3040, 16, 12349}},
    };

    // 用 std::function 包装模板
    map<int, function<void(double *, double *, double *, int, int, int)>> cpu_gemm_versions = {
        {0, cpu_gemm_blas<double>},
        // {1, cpu_gemm_v3<double ,16>},
        // {2, cpu_gemm_v2_1<16>},
        // {2, square_sgemm},
        // {3, cpu_gemm_v4<double>},
        // {3, cpu_gemm_v5<double, 16>},
        // {3, cpu_gemm_v1<double,64,64>},
        // {4, cpu_gemm_v2<double, 16, 16>},
        // {5, cpu_gemm_v2_1<double>},
        {6, cpu_gemm_v2_2<double>},
        {7, cpu_gemm_v3<double>},
    };
    test_gemm(cpu_gemm_versions);
    for (const auto &[key, size] : matrix_size)
    {
        int M = get<0>(size);
        int N = get<1>(size);
        int K = get<2>(size);

        cout << "Matrix size: " << M << "x" << N << "x" << K << endl;

        double* A = (double*)aligned_alloc(64, sizeof(double) * M * K);
        double* B = (double*)aligned_alloc(64, sizeof(double) * K * N);
        double* C = (double*)aligned_alloc(64, sizeof(double) * M * N);
        // Initialize matrices A and B with random values
        for (int i = 0; i < M * K; ++i)
        {
            A[i] = static_cast<double>(rand()) / RAND_MAX;
        }
        for (int i = 0; i < K * N; ++i)
        {
            B[i] = static_cast<double>(rand()) / RAND_MAX;
        }

        for (const auto &[version, gemm_func] : cpu_gemm_versions)
        {   
            cout << "Running CPU GEMM version " << version << endl;
            double total_time = 0.0;
            for(int i=0; i < 10; ++i) // Repeat 10 times for averaging
            {
                memset(C, 0, M * N * sizeof(double));
                auto start = chrono::high_resolution_clock::now();
                gemm_func(A, B, C, M, N, K);
                auto end = chrono::high_resolution_clock::now();
                double elapsed_time = chrono::duration<double>(end - start).count();
                total_time += elapsed_time;          
            }
            double average_time = total_time / 10.0;
            double GFLOPS = (2.0 * M * N * K) / (average_time * 1e9);
            cout << "size " << M << "x" << N << "x" << K
                 << ", version " << version
                 << ", time: " << average_time << " seconds"
                 << ", GFLOPS: " << GFLOPS << endl;
        }

        // Clean up
        delete[] A;
        delete[] B;
        delete[] C;
        cout << "----------------------------------------" << endl;
    }
}
#endif // compute