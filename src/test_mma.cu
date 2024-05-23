/**
 * @file test_mma.cu
 * @author sablin39
 * @brief C=AB+D
 * @version 0.1
 * @date 2024-05-23
 * 
 */

#include <cuda.h>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/atom/mma_atom.hpp>
#include <stdio.h>
#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

#include <type_traits>

template <typename T>
void gen_rand_data(T *data, int n);

template<typename TC, typename TA, typename TB, typename TD, typename TiledMMA>
__global__ void
test_gemm(TC *C, TA *A, TB *B, TD *D , int m, int n, int k) {
    using namespace cute;
    auto tensorA = make_tensor(make_gmem_ptr(A), make_shape(m,k), make_stride(k, Int<1>{}));
    auto tensorB = make_tensor(make_gmem_ptr(B), make_shape(n,k), make_stride(k, Int<1>{}));
    auto tensorC = make_tensor(make_gmem_ptr(C), make_shape(m,n), make_stride(Int<1>{}, m));
    auto tensorD = make_tensor(make_gmem_ptr(D), make_shape(m,n), make_stride(Int<1>{}, m));

    int idx_x = blockIdx.x;
    int idx_y = blockIdx.y;

    int thrIdx_x = threadIdx.x;

    auto bM = Int<64>{};
    auto bN = Int<64>{};
    auto bK = Int<8>{};

    auto cta_tiler = make_shape(bM, bN, bK);
    auto cta_coord = make_coord(idx_x, idx_y, _);

    auto gA = local_tile(tensorA, cta_tiler, cta_coord, Step<_1, X, _1>{});
    auto gB = local_tile(tensorB, cta_tiler, cta_coord, Step<X, _1, _1>{});
    auto gC = local_tile(tensorC, cta_tiler, cta_coord, Step<_1, _1, X>{});
    auto gD = local_tile(tensorD, cta_tiler, cta_coord, Step<_1, _1, X>{});

    auto sA_layout = make_layout(make_shape (      bM,          bK),
                                 make_stride(Int<1>{}, bM+Int<8>{}));
    auto sB_layout = make_layout(make_shape (      bN,          bK),
                                 make_stride(Int<1>{}, bN+Int<8>{}));
    auto sC_layout = make_layout(make_shape(bM, bN));

    __shared__ TA smemA[cosize_v<decltype(sA_layout)>];
    __shared__ TB smemB[cosize_v<decltype(sB_layout)>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);
    TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<TA>, TA>{},
                                      Layout<Shape<_32, _8>,Stride<_8, _1>>{},
                                      Layout<Shape< _4, _1>>{});
    TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<TB>, TB>{},
                                      Layout<Shape<_32, _8>,Stride<_8, _1>>{},
                                      Layout<Shape< _4, _1>>{});
    ThrCopy thr_copy_a = copyA.get_slice(thrIdx_x);
    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tAsA = thr_copy_a.partition_D(sA);
    Tensor tArA = make_fragment_like(tAsA);
    ThrCopy thr_copy_b = copyB.get_slice(thrIdx_x);
    Tensor tBgB = thr_copy_b.partition_S(gB);
    Tensor tBsB = thr_copy_b.partition_D(sB);
    Tensor tBrB = make_fragment_like(tBsB);

    copy(copyA, tAgA(_,_,_,0), tArA);
    copy(copyB, tBgB(_,_,_,0), tBrB);

    TiledMMA mmaC;
    ThrMMA thr_mma = mmaC.get_slice(thrIdx_x);
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCgC = thr_mma.partition_C(gC);
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    Tensor tDgD = thr_mma.partition_C(gD);
    clear(tCrC);

    auto K_TILE_MAX = size<3>(tAgA);

    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
        __syncthreads();
        copy(tArA, tAsA);
        copy(tBrB, tBsB);
        __syncthreads();

        int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
        copy(copyA, tAgA(_,_,_,k_tile_next), tArA);
        copy(copyB, tBgB(_,_,_,k_tile_next), tBrB);

        gemm(mmaC, tCsA, tCsB, tCrC);
    }
    axpby(1.0, tCrC, 1.0, tDgD);

}

int main() {
    int m = 5120;

    int n = 5120;

    int k = 4096;

    using Type = float;

    cute::device_init(0);

    thrust::host_vector<Type> h_A(m*k);
    thrust::host_vector<Type> h_B(n*k);
    thrust::host_vector<Type> h_C(m*n);
    thrust::host_vector<Type> h_D(m*n);

    gen_rand_data(thrust::raw_pointer_cast(h_A.data()), m*k);

    gen_rand_data(thrust::raw_pointer_cast(h_B.data()), n*k);

    gen_rand_data(thrust::raw_pointer_cast(h_D.data()), m*n);
    
    thrust::device_vector<Type> A(m*k);
    thrust::device_vector<Type> B(n*k);
    thrust::device_vector<Type> C(m*n);
    thrust::device_vector<Type> D(m*n);
    
    thrust::copy(h_A.begin(), h_A.end(), A.begin());
    thrust::copy(h_B.begin(), h_B.end(), B.begin());
    thrust::copy(h_D.begin(), h_D.end(), D.begin());

    using mma_op = cute::SM80_16x8x8_F32TF32TF32F32_TN;
    using mma_traits = cute::MMA_Traits<mma_op>;
    using mma_atom = cute::MMA_Atom<mma_traits>;
    using namespace cute;
    using MMA = decltype(make_tiled_mma(mma_atom{},
                         Layout<Shape<_1, _2, _1>>{},
                         Tile<_1, _1, _1>{}));
    dim3 block(128);
    dim3 grid(80, 80);
    double gflops = (2.0*m*n*k) * 1e-9;
    bool printflag = false;
    GPU_Clock gpu_clock;
    for (int iteration = 0; iteration < 10; iteration++) {
        gpu_clock.start();
        test_gemm<Type, Type, Type, Type, MMA><<<grid, block, 0, 0>>>(thrust::raw_pointer_cast(C.data()), 
                                   thrust::raw_pointer_cast(A.data()), 
                                   thrust::raw_pointer_cast(B.data()),
                                   thrust::raw_pointer_cast(D.data()), 
                                   m, n, k);
        CUTE_CHECK_LAST();
        double cute_time = gpu_clock.seconds();
        printf("CUTLASS time: %f ms, GFLOPS: %f\n", cute_time * 1e3, gflops / cute_time);
        thrust::host_vector<Type> C_result = C;
        // if (!printflag) {
        //     for (int i = 0; i < m; i++) {
        //         for (int j = 0; j < n; j++) {
        //             printf("%f ", C_result[i*n+j]);
        //         }
        //         printf("\n");
        //     }
        //     printflag = true;
        // }
    }
    

    return 0;

}

template <typename T>
void gen_rand_data(T *data, int n) {
  for (int i = 0; i < n; ++i) {
    float v = (rand() % 200 - 100) * 0.1;
    data[i] = v;
  }
}