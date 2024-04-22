#include <cuda_runtime_api.h>
#include <cute/numeric/integral_constant.hpp>
#include <cute/config.hpp>
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <vector_types.h>
#include <utility>

namespace cute {
  template <class M = _128, class N = _64, class K = _64, int N_thread = 256,
            int N_pipe = 3>
  struct GemmConfigTF32KMajor {
    using M_128Bit = Int<M::value / 4>;
    using N_128Bit = Int<N::value / 4>;
    using K_128Bit = Int<K::value / 4>;
    static_assert(M_128Bit::value * 4 == M::value);
    static_assert(N_128Bit::value * 4 == N::value);
    static_assert(K_128Bit::value * 4 == K::value);

    static constexpr int Nwarp = N_thread / 32;
    static constexpr int NcopyPerThreadA =
        M_128Bit::value * K_128Bit::value / N_thread;
    static constexpr int NcopyPerThreadB =
        N_128Bit::value * K_128Bit::value / N_thread;
    static constexpr int Npipe = N_pipe;
    static constexpr int Nthread = N_thread;

    using CtaTiler = Shape<M, N, K>;
    using CopyAtom_G2S =
        Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS<float4>>, float4>;

    // DEFINE FOR A
    using SmemLayoutA = Layout<Shape<M, K, Int<Npipe>>,
                               Stride<K, _1, Int<M::value * K::value>>>;
    using SmemLayoutA_128Bit =
        Layout<Shape<M_128Bit, K_128Bit, Int<Npipe>>,
               Stride<K_128Bit, _1, Int<M_128Bit::value * K_128Bit::value>>>;
    using SwizzleA = Swizzle<3, 2, 4>;
    using SwizzleA_128Bit = Swizzle<3, 0, 4>;
    using SmemLayoutA_128Bit_Swizzled =
        decltype(composition(SwizzleA_128Bit{}, SmemLayoutA_128Bit{}));
    using SmemLayoutA_Swizzled =
        decltype(composition(SwizzleA{}, SmemLayoutA{}));
    using CopyTVLayoutA_128Bit_G2S =
        Layout<Shape<Shape<_16, _16>, _8>, Stride<Stride<_128, _1>, _16>>;
    using TiledCopy_G2S_A_128Bit =
        TiledCopy<CopyAtom_G2S, CopyTVLayoutA_128Bit_G2S, Shape<_128, _16>>;

    // DEFINE FOR B
    using SmemLayoutB = Layout<Shape<K, N, Int<Npipe>>,
                               Stride<_1, K, Int<K::value * N::value>>>;
    using SmemLayoutB_128Bit =
        Layout<Shape<K_128Bit, N_128Bit, Int<Npipe>>,
               Stride<_1, K_128Bit, Int<K_128Bit::value * N_128Bit::value>>>;
    using SwizzleB = Swizzle<3, 2, 4>; // 2 means 128 bit
    using SwizzleB_128Bit = Swizzle<3, 0, 4>;
    using SmemLayoutB_128Bit_Swizzled =
        decltype(composition(SwizzleB_128Bit{}, SmemLayoutB_128Bit{}));
    using SmemLayoutB_Swizzled =
        decltype(composition(SwizzleB{}, SmemLayoutB{}));
    using CopyLayoutB_128Bit_G2S = Layout<Shape<_256, _4>, Stride<_1, _256>>;
    using TiledCopy_G2S_B_128Bit =
        TiledCopy<CopyAtom_G2S, CopyLayoutB_128Bit_G2S, Shape<_16, _128>>;
  };
} // namespace cute

template <class Engine_, class Layout_>
CUTE_HOST_DEVICE decltype(auto)
cast_32bit_to_128bit(cute::Tensor<Engine_, Layout_> &&tensor) {
  using namespace cute;
  using LayoutIn = decltype(tensor.layout());
  using ShapeIn = decltype(tensor.shape());
  using StrideIn = decltype(tensor.stride());
  using Iterator = decltype(tensor.data());
  if constexpr (get<0>(StrideIn{}) > get<1>(StrideIn{})) {
    // row major
    using ShapeOut = decltype(make_shape(
        get<0>(ShapeIn{}), get<1>(ShapeIn{}) / 4, get<2>(ShapeIn{})));
    using StrideOut = decltype(make_stride(
        get<0>(StrideIn{}) / 4, get<1>(StrideIn{}), get<2>(StrideIn{}) / 4));
    using LayoutOut = Layout<ShapeOut, StrideOut>;
    if constexpr (is_smem<Iterator>::value) {
      return make_tensor(
          make_smem_ptr(reinterpret_cast<float4 *>(tensor.data())),
          LayoutOut{});
    } else {
      // CUTE_STATIC_ASSERT(is_gmem<Iterator>::value);
      return make_tensor(
          make_gmem_ptr(reinterpret_cast<float4 *>(tensor.data())),
          LayoutOut{});
    }
  } else {
    // col major
    using ShapeOut = decltype(make_shape(
        get<0>(ShapeIn{}) / 4, get<1>(ShapeIn{}), get<2>(ShapeIn{}) / 4));
    using StrideOut = decltype(make_stride(
        get<0>(StrideIn{}), get<1>(StrideIn{}) / 4, get<2>(StrideIn{}) / 4));
    using LayoutOut = Layout<ShapeOut, StrideOut>;
    if constexpr (is_smem<Iterator>::value) {
      return make_tensor(
          make_smem_ptr(reinterpret_cast<float4 *>(tensor.data())),
          LayoutOut{});
    } else {
      // CUTE_STATIC_ASSERT(is_gmem<Iterator>::value);
      return make_tensor(
          make_gmem_ptr(reinterpret_cast<float4 *>(tensor.data())),
          LayoutOut{});
    }
  }
}

template <class T = float, class ProblemShape = cute::Shape<int, int, int>,
          class AStride, class BStride, class CStride,
          class GemmConfig = cute::GemmConfigTF32KMajor<>>
__global__ static void gemm_device(ProblemShape shape_MNK, T const *A,
                                   AStride dA, T const *B, BStride dB, T *C,
                                   CStride dC) {
  using namespace cute;
  using config = GemmConfigTF32KMajor<>;
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB);
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC);
  auto block_coord = make_coord(blockIdx.x, blockIdx.y, _);

  auto cta_tiler = config::CtaTiler{};
  constexpr int bM = get<0>(cta_tiler);
  constexpr int bN = get<1>(cta_tiler);
  constexpr int bK = get<2>(cta_tiler);
  constexpr int smem_A_numel = bM * bK * config::Npipe;
  constexpr int smem_B_numel = bK * bN * config::Npipe;

  // (bM, bK, nk)
  Tensor gA =
      local_tile(mA, select<0, 2>(cta_tiler), select<0, 2>(block_coord));
  // (bK, bN, nk)
  Tensor gB =
      local_tile(mB, select<1, 2>(cta_tiler), select<1, 2>(block_coord));
  // (bM, bN)
  Tensor gC =
      local_tile(mC, select<0, 1>(cta_tiler), select<0, 1>(block_coord));

  auto gA_128bit_layout = make_layout(get<0>(gA.layout()) / 4, get<1>(gA.layout()),
                                      get<2>(gA.layout()) / 4);

  Tensor gA_128bit = make_tensor(make_gmem_ptr(gA.data()), gA.layout());
  Tensor gB_128bit = make_tensor(make_gmem_ptr(gA.data()), gA.layout());

  extern __shared__ float shared_mem_ptr[];
  float *smemA_ptr = shared_mem_ptr;
  float *smemB_ptr = shared_mem_ptr + smem_A_numel;
  float4 *smemA_ptr_128bit = reinterpret_cast<float4 *>(smemA_ptr);
  float4 *smemB_ptr_128bit = reinterpret_cast<float4 *>(smemB_ptr);

  Tensor smemA_tensor_128bit_swizzeled = make_tensor(
      make_smem_ptr(smemA_ptr_128bit), config::SmemLayoutA_128Bit_Swizzled{});
  Tensor smemB_tensor_128bit_swizzeled = make_tensor(
      make_smem_ptr(smemB_ptr_128bit), config::SmemLayoutB_128Bit_Swizzled{});

  config::TiledCopy_G2S_A_128Bit copyA_g2s{};
  config::TiledCopy_G2S_B_128Bit copyB_g2s{};
  auto thr_copy_A_g2s = copyA_g2s.get_slice(threadIdx.x);
  auto thr_copy_B_g2s = copyB_g2s.get_slice(threadIdx.x);
  auto thr_tensor_A_128bit_g = thr_copy_A_g2s.partition_S(gA_128bit);
  auto thr_tensor_B_128bit_g = thr_copy_B_g2s.partition_S(gB_128bit);
  auto thr_tensor_A_128bit_s =
      thr_copy_A_g2s.partition_D(smemA_tensor_128bit_swizzeled);
  auto thr_tensor_B_128bit_s =
      thr_copy_B_g2s.partition_D(smemB_tensor_128bit_swizzeled);
}

__global__ void test_cute_host() {
  using namespace cute;
  float *ptr = new float[256];
  for (int i = 0; i < 256; ++i) {
    ptr[i] = i;
  }
  using LayoutA =
      decltype(make_layout(Shape<_16, _8, _2>{}, Stride<_8, _1, _128>{}));
  Tensor tensor = make_tensor(ptr, LayoutA{});
  Tensor cast_t = cast_32bit_to_128bit(std::move(tensor));
  using CopyAtom_G2S =
      Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS<float4>>, float4>;
  using CopyLayoutB_128Bit_G2S = Layout<Shape<_256, _4>, Stride<_1, _256>>;
  using TiledCopy_G2S_B_128Bit =
      TiledCopy<CopyAtom_G2S, CopyLayoutB_128Bit_G2S, Shape<_16, _64>>;
  // TiledCopy_G2S_B_128Bit copy_t{};
  // print_latex(copy_t);
  // print(cast_t.layout());
  // print_tensor(cast_t);
  // print_layout(coalesce(cast_t.layout()));
}

int main() {
    
  test_cute_host<<<1,1>>>();
  cudaDeviceSynchronize();
  return 0;
}

