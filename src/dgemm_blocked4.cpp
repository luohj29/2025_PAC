#include <immintrin.h>
#include <algorithm>
#include <vector>
#include <iostream>

namespace block3{
    // A_stride_k: The distance in memory between A[i][j] and A[i+1][j]. This is the K of the original matrix.
    // K: The number of columns in the A-tile we are processing (the k-dimension of the block).
    template <typename T>
    inline void micro_kernel(const T *A, const T *B, T *C,
                            int N_stride, int K, int A_stride_k) // <<< CHANGED SIGNATURE
    {
        __m256d c_reg_0 = _mm256_setzero_pd();
        __m256d c_reg_1 = _mm256_setzero_pd();
        __m256d c_reg_2 = _mm256_setzero_pd();
        __m256d c_reg_3 = _mm256_setzero_pd();

        for (int k = 0; k < K; k++)
        {
            __m256d b_reg = _mm256_loadu_pd(&B[k * N_stride]);

            // *** THE FIX IS HERE ***
            // Use the original matrix stride (A_stride_k) to move between rows of A.
            __m256d a_reg_0 = _mm256_broadcast_sd(&A[0 * A_stride_k + k]);
            __m256d a_reg_1 = _mm256_broadcast_sd(&A[1 * A_stride_k + k]);
            __m256d a_reg_2 = _mm256_broadcast_sd(&A[2 * A_stride_k + k]);
            __m256d a_reg_3 = _mm256_broadcast_sd(&A[3 * A_stride_k + k]);

            c_reg_0 = _mm256_fmadd_pd(a_reg_0, b_reg, c_reg_0);
            c_reg_1 = _mm256_fmadd_pd(a_reg_1, b_reg, c_reg_1);
            c_reg_2 = _mm256_fmadd_pd(a_reg_2, b_reg, c_reg_2);
            c_reg_3 = _mm256_fmadd_pd(a_reg_3, b_reg, c_reg_3);
        }

        _mm256_storeu_pd(&C[0 * N_stride], _mm256_add_pd(_mm256_loadu_pd(&C[0 * N_stride]), c_reg_0));
        _mm256_storeu_pd(&C[1 * N_stride], _mm256_add_pd(_mm256_loadu_pd(&C[1 * N_stride]), c_reg_1));
        _mm256_storeu_pd(&C[2 * N_stride], _mm256_add_pd(_mm256_loadu_pd(&C[2 * N_stride]), c_reg_2));
        _mm256_storeu_pd(&C[3 * N_stride], _mm256_add_pd(_mm256_loadu_pd(&C[3 * N_stride]), c_reg_3));
    }

    // M, N, K are the dimensions of the block.
    // A_stride_k is the row stride of the ORIGINAL A matrix.
    template <typename T, int tiley = 4, int tilex = 4>
    inline void inner_kernel(const T *A, const T *B, T *C,
                            int M, int N, int K, int A_stride_k) // <<< CHANGED SIGNATURE
    {
        for(int ty = 0; ty < M; ty += tiley){
            for(int tx = 0; tx < N; tx += tilex){
                int my_tile = std::min(tiley, M - ty);
                int nx_tile = std::min(tilex, N - tx);

                if (my_tile == tiley && nx_tile == tilex) {
                    // *** THE FIX IS HERE ***
                    // Calculate the starting pointer for the A-tile using the correct stride.
                    const T* a_tile_ptr = &A[ty * A_stride_k];
                    // Pass all strides down to the micro-kernel.
                    micro_kernel(a_tile_ptr, &B[tx], &C[ty * N + tx], N, K, A_stride_k);
                }
                else {
                    // Scalar code also needs to use the correct stride.
                    for (int i = 0; i < my_tile; ++i) {
                        for (int j = 0; j < nx_tile; ++j) {
                            T sum = 0;
                            for (int k = 0; k < K; ++k) {
                                // Use A_stride_k to find the correct row in A
                                sum += A[(ty + i) * A_stride_k + k] * B[k * N + (tx + j)];
                            }
                            C[(ty + i) * N + (tx + j)] += sum;
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
inline void packA(const T *dst, const T *A, int M, int K, int tiley=4){
    // This function is a placeholder for packing A into a contiguous memory layout.
    // In practice, you would implement this to optimize memory access patterns.
    // For now, we assume A is already in a suitable format.
    // You can use std::vector or similar to create a packed version if needed.
    for (int y = 0; y < M; y+=tiley) {
        for (int x = 0; x < K; x++) {
          for (int iy = 0; iy < tiley; iy++) {
            dst[y*K + x*tiley + iy] = A[(y+iy)*K + x];
          }
        }
    }
}

template <typename T, int blocky = 32, int blockk = 32>
inline void cpu_gemm_v3(const T *A, const T *B, T *C,
                          int M, int N, int K)
{
    // It's good practice to ensure C is zeroed out before accumulation.
    for (int i = 0; i < M * N; ++i) {
        C[i] = 0;
    }

    T* packedA = (T*)aligned_alloc(64, sizeof(T) * blocky * blockk);
    for (int by = 0; by < M; by += blocky)
    {
        int my = std::min(blocky, M - by);
        for (int bk = 0; bk < K; bk += blockk)
        {
            int mk = std::min(blockk, K - bk);

            //TODO: PACC tha A data to contiguous memory
            packA(packedA, &A[by * K + bk], my, mk, blocky);
            // *** THE FIX IS HERE ***
            // Pass the original K as the A_stride_k parameter.
            block3::inner_kernel<T>(
                packedA,            // Pointer to start of packed A block
                &B[bk * N],         // Pointer to start of B block
                &C[by * N],         // Pointer to start of C block
                my,                 // M dimension of block
                N,                  // N dimension is the full N
                mk,                 // K dimension of block
                K                   // The real row stride of A
            );
        }
    }
}