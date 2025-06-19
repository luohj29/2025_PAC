/*basic 2d tiling*/
template <typename T,  int blocky = 64, int blockx = 64>
inline void cpu_gemm_v1(const T* A, const T* B, T* C,
                        int M, int N, int K) {
    for(int by = 0; by < M; by += blocky) {
        for(int bx = 0; bx < N; bx += blockx) {
            T tc[blocky][blockx] = {0};
            for(int k=0; k<K; k++){
                for(int y=0; y<blocky; y++){
                    for(int x =0; x<blockx; x++){
                        tc[y][x] += A[(by+y)*K + k] * B[k*N + (bx+x)];
                    }
                }
            }

            //store
            for(int y=0; y<blocky; y++){
                for(int x =0; x<blockx; x++){
                    C[(by+y)*N + (bx+x)] = tc[y][x];
                }
            }
        }
    }
}