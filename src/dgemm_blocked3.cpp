#include <immintrin.h>
#include <algorithm>

// use multi-blocks to process the matrix
// outter-kernel make A blocky x blockk, B blockk x N
// inner-kernel make A blocky x blockk, B blockk x N
namespace block3{
    template <typename T>
    inline void micro_kernel(const T *A, const T *B, T *C,
                            int N, int K)
    { // make 4x4 to C
        __m256d c_reg_0 = {}, c_reg_1 = {}, c_reg_2 = {}, c_reg_3 = {};

        for (int k = 0; k < K; k++)
        {
            __m256d b_reg = _mm256_loadu_pd(&B[k * N]);
            __m256d a_reg_0 = _mm256_broadcast_sd(&A[0 * K + k]);
            __m256d a_reg_1 = _mm256_broadcast_sd(&A[1 * K + k]);
            __m256d a_reg_2 = _mm256_broadcast_sd(&A[2 * K + k]);
            __m256d a_reg_3 = _mm256_broadcast_sd(&A[3 * K + k]);
        
            c_reg_0 = _mm256_fmadd_pd(a_reg_0, b_reg, c_reg_0);
            c_reg_1 = _mm256_fmadd_pd(a_reg_1, b_reg, c_reg_1);
            c_reg_2 = _mm256_fmadd_pd(a_reg_2, b_reg, c_reg_2);
            c_reg_3 = _mm256_fmadd_pd(a_reg_3, b_reg, c_reg_3);
        }

        _mm256_storeu_pd(&C[0], _mm256_add_pd(_mm256_loadu_pd(&C[0]), c_reg_0));  //you need to accumulate the results, instead of overwrite
        _mm256_storeu_pd(&C[N], _mm256_add_pd(_mm256_loadu_pd(&C[N]), c_reg_1));
        _mm256_storeu_pd(&C[N * 2], _mm256_add_pd(_mm256_loadu_pd(&C[N * 2]), c_reg_2));
        _mm256_storeu_pd(&C[N * 3], _mm256_add_pd(_mm256_loadu_pd(&C[N * 3]), c_reg_3));
    }

    template <typename T, int blocky = 4, int blockk = 4>
    inline void inner_kernel(const T *A, const T *B, T *C,
                            int M, int N, int K)
    {
        for(int by = 0; by < M; by += blocky){
            for(int bx = 0; bx < N; bx += blockk){
                int my = std::min(blocky, M - by);
                int nx = std::min(blockk, N - bx);
                if (my == blocky && nx == blockk) {
                    micro_kernel(&A[by * K], &B[bx], &C[by * N + bx], N, K);
                } else {
                    // 边界处理（可选：用标量代码或补零/补齐）
                    for (int i = 0; i < my; ++i)
                        for (int j = 0; j < nx; ++j) {
                            T sum = 0;
                            for (int k = 0; k < K; ++k)
                                sum += A[(by + i) * K + k] * B[k * N + (bx + j)];
                            C[(by + i) * N + (bx + j)] += sum;
                        }
                }
            }
        }
    }
}

template <typename T, int blocky = 32, int blockk = 32>
inline void cpu_gemm_v3(const T *A, const T *B, T *C,
                          int M, int N, int K)
{
    for (int by = 0; by < M; by += blocky)
    {
        int my = std::min(blocky, M - by);
        for (int bk = 0; bk < K; bk += blockk)
        {
            int mk = std::min(blockk, K - bk);
            block3::inner_kernel<T>(
                &A[by * K + bk],
                &B[bk * N],
                &C[by * N],
                my, N, mk
            );
        }
    }
}