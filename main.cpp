#include <benchmark/benchmark.h>
#include "tensor-ops/NaiveOperations.h"
#include "tensor-ops/EigenOperations.h"
#include "tensor-ops/ExperimentalOperations.h"
#include "tensor-ops/NativeOperations.h"


constexpr int tensor_size=100;
//
// Native Tensor Library - Loop Unrolled
// static void BM_Native_TensorMultiply_Unrolled(benchmark::State& state) {
//     NativeTensorWrapper A(tensor_size, tensor_size, tensor_size);
//     NativeTensorWrapper B(tensor_size, tensor_size, tensor_size);
//     NativeTensorWrapper C(tensor_size, tensor_size, tensor_size);
//
//     A.randomize(-100.0f, 100.0f);
//     B.randomize(-100.0f, 100.0f);
//
//     for (auto _ : state) {
//         A.multiplyElementwiseUnrolled(B, C);
//         benchmark::DoNotOptimize(C);
//         benchmark::ClobberMemory();
//     }
//
//     int total_ops = tensor_size * tensor_size * tensor_size;
//     state.SetItemsProcessed(state.iterations() * total_ops);
// }
// BENCHMARK(BM_Native_TensorMultiply_Unrolled)
//     ->Unit(benchmark::kMicrosecond)
//     ->MinWarmUpTime(2.0)
//     ->Repetitions(5)
//     ->ReportAggregatesOnly();

//
// static void BM_ETensor_TensorMultiply_SPECIAL(benchmark::State& state) {
//     // For simplicity, use all dimensions equal: M = K = N = D = tensor_size
//     const int M = tensor_size;
//     const int K = tensor_size;
//     const int N = tensor_size;
//     const int D = tensor_size;
//
//     ETensorWrapper A(M, K, D);
//     ETensorWrapper B(K, N, D);
//     ETensorWrapper C(M, N, D);
//
//     A.randomize(-100.0f, 100.0f);
//     B.randomize(-100.0f, 100.0f);
//
//     for (auto _ : state) {
//         A.multiplyMatmulSpecial(B, C);
//         benchmark::DoNotOptimize(C);
//         benchmark::ClobberMemory();
//     }
//
//
// }
//
// BENCHMARK(BM_ETensor_TensorMultiply_SPECIAL)
//     ->Unit(benchmark::kMicrosecond)
//     ->MinWarmUpTime(2.0)
//     ->Repetitions(5)
//     ->ReportAggregatesOnly();
//
// static void BM_Native_TensorMultiply_SIMD(benchmark::State& state) {
//     NativeTensorWrapper A(tensor_size, tensor_size, tensor_size);
//     NativeTensorWrapper B(tensor_size, tensor_size, tensor_size);
//     NativeTensorWrapper C(tensor_size, tensor_size, tensor_size);
//
//     A.randomize(-100.0f, 100.0f);
//     B.randomize(-100.0f, 100.0f);
//
//     for (auto _ : state) {
//         A.multiplyElementwiseSimd(B, C);
//         benchmark::DoNotOptimize(C);
//         benchmark::ClobberMemory();
//     }
//
//     int total_ops = tensor_size * tensor_size * tensor_size;
//     state.SetItemsProcessed(state.iterations() * total_ops);
// }
// BENCHMARK(BM_Native_TensorMultiply_SIMD)
//     ->Unit(benchmark::kMicrosecond)
//     ->MinWarmUpTime(2.0)
//     ->Repetitions(5)
//     ->ReportAggregatesOnly();

//
// // Eigen implementation
// static void BM_Eigen_TensorMultiply(benchmark::State& state) {
//     EigenTensor A(tensor_size, tensor_size, tensor_size);
//     EigenTensor B(tensor_size, tensor_size, tensor_size);
//     EigenTensor C(tensor_size, tensor_size, tensor_size);
//
//     A.randomize(-100.0f, 100.0f);
//     B.randomize(-100.0f, 100.0f);
//
//     for (auto _ : state) {
//         A.multiply(B, C);
//         benchmark::DoNotOptimize(C);
//         benchmark::ClobberMemory();
//     }
//
//     int total_ops = tensor_size * tensor_size * tensor_size;
//     state.SetItemsProcessed(state.iterations() * total_ops);
// }
// BENCHMARK(BM_Eigen_TensorMultiply)
//     ->Unit(benchmark::kMicrosecond)
//     ->MinWarmUpTime(2.0)
//     ->Repetitions(5)
//     ->ReportAggregatesOnly();
static void BM_Eigen_TensorMultiply(benchmark::State& state) {
    const int M = tensor_size;
    const int K = tensor_size;
    const int N = tensor_size;
    const int D = tensor_size;

    EigenTensor A(M, K, D);
    EigenTensor B(K, N, D);
    EigenTensor C(M, N, D);

    A.randomize(-100.0f, 100.0f);
    B.randomize(-100.0f, 100.0f);

    for (auto _ : state) {
        A.multiply_GEMM(B, C);  // now GEMM per depth
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory();
    }
    const std::int64_t mul_adds_per_iter =
    static_cast<std::int64_t>(M) * K * N * D * 2; // *2 for Multiply + Add

    state.SetItemsProcessed(state.iterations() * mul_adds_per_iter);

}

BENCHMARK(BM_Eigen_TensorMultiply)
    ->Unit(benchmark::kMicrosecond)
    ->MinWarmUpTime(2.0)
    ->Repetitions(5)
    ->ReportAggregatesOnly();

static void BM_ETensor_TensorMultiply_SIMD(benchmark::State& state) {
    const int M = tensor_size;
    const int K = tensor_size;
    const int N = tensor_size;
    const int D = tensor_size;

    ETensorWrapper A(M, K, D);
    ETensorWrapper B(K, N, D);
    ETensorWrapper C(M, N, D);

    A.randomize(-100.0f, 100.0f);
    B.randomize(-100.0f, 100.0f);

    for (auto _ : state) {
        A.multiplyMatmulSimd(B, C);
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory();
    }

    const std::int64_t mul_adds_per_iter =
        static_cast<std::int64_t>(M) * K * N * D * 2;

    state.SetItemsProcessed(state.iterations() * mul_adds_per_iter);
}
BENCHMARK(BM_ETensor_TensorMultiply_SIMD)
    ->Unit(benchmark::kMicrosecond)
    ->MinWarmUpTime(2.0)
    ->Repetitions(5)
    ->ReportAggregatesOnly();



 BENCHMARK_MAIN();