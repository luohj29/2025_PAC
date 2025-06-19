#include <cblas.h>

template <typename T>
inline void cpu_gemm_blas(T* A, T* B, T* C, int M, int N, int K) {
    // 这里只支持 float 类型
    cblas_dgemm(
        CblasRowMajor,    // 矩阵存储方式：行主序
        CblasNoTrans,     // A 不转置
        CblasNoTrans,     // B 不转置
        M, N, K,          // 矩阵维度
        1.0,              // alpha
        A, K,             // A, lda
        B, N,             // B, ldb
        0.0,              // beta
        C, N              // C, ldc
    );
}