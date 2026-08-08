#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kGeneralScatter = 0;
constexpr int kPermutationGather = 1;

#define FLASH_PD_LAUNCH(kernel, grid, block, shared, stream, ...) \
    kernel<<<grid, block, shared, stream>>>(__VA_ARGS__)
#define MAMBA3_PHASE_A(scalar, mode) mamba3_phase_a_kernel<scalar, mode>
#define MAMBA3_PHASE_B(mode) mamba3_phase_b_kernel<mode>
#define MAMBA3_PHASE_C(scalar, mode) mamba3_phase_c_kernel<scalar, mode>

template <typename scalar_t>
__device__ __forceinline__ float as_float(scalar_t value) {
    return static_cast<float>(value);
}

__device__ __forceinline__ void complex_product(
    float ar,
    float ai,
    float br,
    float bi,
    float& out_r,
    float& out_i) {
    out_r = ar * br - ai * bi;
    out_i = ar * bi + ai * br;
}

// Marker and standalone formulation used by source-level no-atomic validation.
__device__ __forceinline__ void permutation_step(
    int output,
    const int* inverse,
    const float* staged_real,
    const float* staged_imag,
    float bias_real,
    float bias_imag,
    float& next_real,
    float& next_imag) {
    const int source = inverse[output];
    next_real = bias_real + staged_real[source];
    next_imag = bias_imag + staged_imag[source];
}

__device__ __forceinline__ void scatter_step(
    int destination,
    float value_real,
    float value_imag,
    float* output_real,
    float* output_imag) {
    const unsigned active = __activemask();
    const unsigned peers = __match_any_sync(active, destination);
    float aggregate_real = 0.0f;
    float aggregate_imag = 0.0f;
    unsigned remaining = peers;
    while (remaining != 0) {
        const int source_lane = __ffs(remaining) - 1;
        const float peer_real = __shfl_sync(active, value_real, source_lane);
        const float peer_imag = __shfl_sync(active, value_imag, source_lane);
        aggregate_real += peer_real;
        aggregate_imag += peer_imag;
        remaining &= remaining - 1;
    }
    const int warp_lane = threadIdx.x & 31;
    if (warp_lane == __ffs(peers) - 1) {
        atomicAdd(output_real + destination, aggregate_real);
        atomicAdd(output_imag + destination, aggregate_imag);
    }
}

template <typename scalar_t>
__global__ void phase_a_kernel(
    const int16_t* __restrict__ dictionary_destination,
    const int16_t* __restrict__ routes,
    const scalar_t* __restrict__ diagonal_real,
    const scalar_t* __restrict__ diagonal_imag,
    const scalar_t* __restrict__ bias_real,
    const scalar_t* __restrict__ bias_imag,
    int16_t* __restrict__ aggregate_destination,
    float* __restrict__ aggregate_diagonal_real,
    float* __restrict__ aggregate_diagonal_imag,
    float* __restrict__ aggregate_bias_real,
    float* __restrict__ aggregate_bias_imag,
    int heads,
    int dictionary_size,
    int time,
    int state,
    int chunks,
    int chunk_size,
    int mode) {
    const int row = blockIdx.x;
    const int chunk = blockIdx.y;
    const int lane = threadIdx.x;
    if (lane >= state) {
        return;
    }
    const int head = row % heads;

    extern __shared__ unsigned char shared_raw[];
    int* shared_map = reinterpret_cast<int*>(shared_raw);
    float* shared_diagonal_real = reinterpret_cast<float*>(shared_map + state);
    float* shared_diagonal_imag = shared_diagonal_real + state;
    float* shared_bias_real = shared_diagonal_imag + state;
    float* shared_bias_imag = shared_bias_real + state;
    float* shared_value_real = shared_bias_imag + state;
    float* shared_value_imag = shared_value_real + state;

    int accumulator_destination = lane;
    float accumulator_diagonal_real = 1.0f;
    float accumulator_diagonal_imag = 0.0f;
    float accumulator_bias_real = 0.0f;
    float accumulator_bias_imag = 0.0f;

    const int chunk_start = chunk * chunk_size;
    const int chunk_end = min(chunk_start + chunk_size, time);
    for (int token = chunk_start; token < chunk_end; ++token) {
        const int route = static_cast<int>(routes[row * time + token]);
        const int dictionary_offset =
            (head * dictionary_size + route) * state;
        const int token_offset = (row * time + token) * state + lane;
        shared_map[lane] =
            static_cast<int>(dictionary_destination[dictionary_offset + lane]);
        shared_diagonal_real[lane] = as_float(diagonal_real[token_offset]);
        shared_diagonal_imag[lane] = as_float(diagonal_imag[token_offset]);
        shared_bias_real[lane] = as_float(bias_real[token_offset]);
        shared_bias_imag[lane] = as_float(bias_imag[token_offset]);
        __syncthreads();

        const int old_destination = accumulator_destination;
        const int next_destination = shared_map[old_destination];
        float next_diagonal_real;
        float next_diagonal_imag;
        complex_product(
            shared_diagonal_real[old_destination],
            shared_diagonal_imag[old_destination],
            accumulator_diagonal_real,
            accumulator_diagonal_imag,
            next_diagonal_real,
            next_diagonal_imag);
        complex_product(
            shared_diagonal_real[lane],
            shared_diagonal_imag[lane],
            accumulator_bias_real,
            accumulator_bias_imag,
            shared_value_real[lane],
            shared_value_imag[lane]);
        __syncthreads();

        float next_bias_real;
        float next_bias_imag;
        if (mode == kPermutationGather) {
            const int destination = shared_map[lane];
            shared_map[destination] = lane;
            __syncthreads();
            permutation_step(
                lane,
                shared_map,
                shared_value_real,
                shared_value_imag,
                shared_bias_real[lane],
                shared_bias_imag[lane],
                next_bias_real,
                next_bias_imag);
        } else {
            const int destination = shared_map[lane];
            scatter_step(
                destination,
                shared_value_real[lane],
                shared_value_imag[lane],
                shared_bias_real,
                shared_bias_imag);
            __syncthreads();
            next_bias_real = shared_bias_real[lane];
            next_bias_imag = shared_bias_imag[lane];
        }
        __syncthreads();

        accumulator_destination = next_destination;
        accumulator_diagonal_real = next_diagonal_real;
        accumulator_diagonal_imag = next_diagonal_imag;
        accumulator_bias_real = next_bias_real;
        accumulator_bias_imag = next_bias_imag;
    }

    const int chunk_offset = (row * chunks + chunk) * state + lane;
    aggregate_destination[chunk_offset] =
        static_cast<int16_t>(accumulator_destination);
    aggregate_diagonal_real[chunk_offset] = accumulator_diagonal_real;
    aggregate_diagonal_imag[chunk_offset] = accumulator_diagonal_imag;
    aggregate_bias_real[chunk_offset] = accumulator_bias_real;
    aggregate_bias_imag[chunk_offset] = accumulator_bias_imag;
}

__global__ void phase_b_kernel(
    const int16_t* __restrict__ aggregate_destination,
    const float* __restrict__ aggregate_diagonal_real,
    const float* __restrict__ aggregate_diagonal_imag,
    const float* __restrict__ aggregate_bias_real,
    const float* __restrict__ aggregate_bias_imag,
    int16_t* __restrict__ prefix_destination,
    float* __restrict__ prefix_diagonal_real,
    float* __restrict__ prefix_diagonal_imag,
    float* __restrict__ prefix_bias_real,
    float* __restrict__ prefix_bias_imag,
    int state,
    int chunks,
    int mode) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    if (lane >= state) {
        return;
    }

    extern __shared__ unsigned char shared_raw[];
    int* shared_map = reinterpret_cast<int*>(shared_raw);
    float* shared_diagonal_real = reinterpret_cast<float*>(shared_map + state);
    float* shared_diagonal_imag = shared_diagonal_real + state;
    float* shared_bias_real = shared_diagonal_imag + state;
    float* shared_bias_imag = shared_bias_real + state;
    float* shared_value_real = shared_bias_imag + state;
    float* shared_value_imag = shared_value_real + state;

    int accumulator_destination = lane;
    float accumulator_diagonal_real = 1.0f;
    float accumulator_diagonal_imag = 0.0f;
    float accumulator_bias_real = 0.0f;
    float accumulator_bias_imag = 0.0f;

    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int chunk_offset = (row * chunks + chunk) * state + lane;
        prefix_destination[chunk_offset] =
            static_cast<int16_t>(accumulator_destination);
        prefix_diagonal_real[chunk_offset] = accumulator_diagonal_real;
        prefix_diagonal_imag[chunk_offset] = accumulator_diagonal_imag;
        prefix_bias_real[chunk_offset] = accumulator_bias_real;
        prefix_bias_imag[chunk_offset] = accumulator_bias_imag;

        shared_map[lane] =
            static_cast<int>(aggregate_destination[chunk_offset]);
        shared_diagonal_real[lane] = aggregate_diagonal_real[chunk_offset];
        shared_diagonal_imag[lane] = aggregate_diagonal_imag[chunk_offset];
        shared_bias_real[lane] = aggregate_bias_real[chunk_offset];
        shared_bias_imag[lane] = aggregate_bias_imag[chunk_offset];
        __syncthreads();

        const int old_destination = accumulator_destination;
        const int next_destination = shared_map[old_destination];
        float next_diagonal_real;
        float next_diagonal_imag;
        complex_product(
            shared_diagonal_real[old_destination],
            shared_diagonal_imag[old_destination],
            accumulator_diagonal_real,
            accumulator_diagonal_imag,
            next_diagonal_real,
            next_diagonal_imag);
        complex_product(
            shared_diagonal_real[lane],
            shared_diagonal_imag[lane],
            accumulator_bias_real,
            accumulator_bias_imag,
            shared_value_real[lane],
            shared_value_imag[lane]);
        __syncthreads();

        float next_bias_real;
        float next_bias_imag;
        if (mode == kPermutationGather) {
            const int destination = shared_map[lane];
            shared_map[destination] = lane;
            __syncthreads();
            permutation_step(
                lane,
                shared_map,
                shared_value_real,
                shared_value_imag,
                shared_bias_real[lane],
                shared_bias_imag[lane],
                next_bias_real,
                next_bias_imag);
        } else {
            const int destination = shared_map[lane];
            scatter_step(
                destination,
                shared_value_real[lane],
                shared_value_imag[lane],
                shared_bias_real,
                shared_bias_imag);
            __syncthreads();
            next_bias_real = shared_bias_real[lane];
            next_bias_imag = shared_bias_imag[lane];
        }
        __syncthreads();

        accumulator_destination = next_destination;
        accumulator_diagonal_real = next_diagonal_real;
        accumulator_diagonal_imag = next_diagonal_imag;
        accumulator_bias_real = next_bias_real;
        accumulator_bias_imag = next_bias_imag;
    }
}

template <typename scalar_t>
__global__ void phase_c_kernel(
    const int16_t* __restrict__ dictionary_destination,
    const int16_t* __restrict__ routes,
    const scalar_t* __restrict__ diagonal_real,
    const scalar_t* __restrict__ diagonal_imag,
    const scalar_t* __restrict__ bias_real,
    const scalar_t* __restrict__ bias_imag,
    const float* __restrict__ prefix_bias_real,
    const float* __restrict__ prefix_bias_imag,
    scalar_t* __restrict__ output_real,
    scalar_t* __restrict__ output_imag,
    int heads,
    int dictionary_size,
    int time,
    int state,
    int chunks,
    int chunk_size,
    int mode) {
    const int row = blockIdx.x;
    const int chunk = blockIdx.y;
    const int lane = threadIdx.x;
    if (lane >= state) {
        return;
    }
    const int head = row % heads;

    extern __shared__ unsigned char shared_raw[];
    int* shared_map = reinterpret_cast<int*>(shared_raw);
    float* shared_diagonal_real = reinterpret_cast<float*>(shared_map + state);
    float* shared_diagonal_imag = shared_diagonal_real + state;
    float* shared_bias_real = shared_diagonal_imag + state;
    float* shared_bias_imag = shared_bias_real + state;
    float* shared_value_real = shared_bias_imag + state;
    float* shared_value_imag = shared_value_real + state;

    const int prefix_offset = (row * chunks + chunk) * state + lane;
    float state_real = prefix_bias_real[prefix_offset];
    float state_imag = prefix_bias_imag[prefix_offset];
    const int chunk_start = chunk * chunk_size;
    const int chunk_end = min(chunk_start + chunk_size, time);

    for (int token = chunk_start; token < chunk_end; ++token) {
        const int route = static_cast<int>(routes[row * time + token]);
        const int dictionary_offset =
            (head * dictionary_size + route) * state;
        const int token_offset = (row * time + token) * state + lane;
        shared_map[lane] =
            static_cast<int>(dictionary_destination[dictionary_offset + lane]);
        shared_diagonal_real[lane] = as_float(diagonal_real[token_offset]);
        shared_diagonal_imag[lane] = as_float(diagonal_imag[token_offset]);
        shared_bias_real[lane] = as_float(bias_real[token_offset]);
        shared_bias_imag[lane] = as_float(bias_imag[token_offset]);
        complex_product(
            shared_diagonal_real[lane],
            shared_diagonal_imag[lane],
            state_real,
            state_imag,
            shared_value_real[lane],
            shared_value_imag[lane]);
        __syncthreads();

        float next_real;
        float next_imag;
        if (mode == kPermutationGather) {
            const int destination = shared_map[lane];
            shared_map[destination] = lane;
            __syncthreads();
            permutation_step(
                lane,
                shared_map,
                shared_value_real,
                shared_value_imag,
                shared_bias_real[lane],
                shared_bias_imag[lane],
                next_real,
                next_imag);
        } else {
            const int destination = shared_map[lane];
            scatter_step(
                destination,
                shared_value_real[lane],
                shared_value_imag[lane],
                shared_bias_real,
                shared_bias_imag);
            __syncthreads();
            next_real = shared_bias_real[lane];
            next_imag = shared_bias_imag[lane];
        }
        __syncthreads();

        state_real = next_real;
        state_imag = next_imag;
        output_real[token_offset] = static_cast<scalar_t>(state_real);
        output_imag[token_offset] = static_cast<scalar_t>(state_imag);
    }
}

template <typename scalar_t>
__global__ void mamba3_preprocess_kernel(
    const int16_t* __restrict__ dictionary_destination,
    const int16_t* __restrict__ inverse_destination,
    const int16_t* __restrict__ routes,
    const scalar_t* __restrict__ diagonal_real,
    const scalar_t* __restrict__ diagonal_imag,
    const scalar_t* __restrict__ value_real,
    const scalar_t* __restrict__ value_imag,
    const scalar_t* __restrict__ beta,
    const scalar_t* __restrict__ gamma,
    scalar_t* __restrict__ bias_real,
    scalar_t* __restrict__ bias_imag,
    int heads,
    int dictionary_size,
    int time,
    int state,
    int mode) {
    const int row = blockIdx.x;
    const int token = blockIdx.y;
    const int lane = threadIdx.x;
    const int head = row % heads;
    const int token_offset = (row * time + token) * state + lane;
    const int coefficient_offset = row * time + token;
    const int route = static_cast<int>(routes[row * time + token]);
    const int dictionary_offset = (head * dictionary_size + route) * state;

    extern __shared__ float shared[];
    float* shared_product_real = shared;
    float* shared_product_imag = shared_product_real + state;
    float* shared_output_real = shared_product_imag + state;
    float* shared_output_imag = shared_output_real + state;
    const float previous_value_real =
        token == 0 ? 0.0f : as_float(value_real[token_offset - state]);
    const float previous_value_imag =
        token == 0 ? 0.0f : as_float(value_imag[token_offset - state]);
    const float token_beta = as_float(beta[coefficient_offset]);
    float product_real;
    float product_imag;
    complex_product(
        as_float(diagonal_real[token_offset]),
        as_float(diagonal_imag[token_offset]),
        token_beta * previous_value_real,
        token_beta * previous_value_imag,
        product_real,
        product_imag);
    shared_product_real[lane] = product_real;
    shared_product_imag[lane] = product_imag;
    const float token_gamma = as_float(gamma[coefficient_offset]);
    shared_output_real[lane] =
        token_gamma * as_float(value_real[token_offset]);
    shared_output_imag[lane] =
        token_gamma * as_float(value_imag[token_offset]);
    __syncthreads();

    if (mode == kPermutationGather) {
        const int source =
            static_cast<int>(inverse_destination[dictionary_offset + lane]);
        shared_output_real[lane] += shared_product_real[source];
        shared_output_imag[lane] += shared_product_imag[source];
    } else {
        const int destination =
            static_cast<int>(dictionary_destination[dictionary_offset + lane]);
        scatter_step(
            destination,
            shared_product_real[lane],
            shared_product_imag[lane],
            shared_output_real,
            shared_output_imag);
    }
    __syncthreads();
    bias_real[token_offset] =
        static_cast<scalar_t>(shared_output_real[lane]);
    bias_imag[token_offset] =
        static_cast<scalar_t>(shared_output_imag[lane]);
}

template <typename scalar_t, int mode>
__global__ void mamba3_phase_a_kernel(
    const int16_t* __restrict__ dictionary_destination,
    const int16_t* __restrict__ inverse_destination,
    const int16_t* __restrict__ routes,
    const scalar_t* __restrict__ diagonal_real,
    const scalar_t* __restrict__ diagonal_imag,
    const scalar_t* __restrict__ value_real,
    const scalar_t* __restrict__ value_imag,
    const scalar_t* __restrict__ beta,
    const scalar_t* __restrict__ gamma,
    int16_t* __restrict__ aggregate_destination,
    float* __restrict__ aggregate_diagonal_real,
    float* __restrict__ aggregate_diagonal_imag,
    float* __restrict__ aggregate_bias_real,
    float* __restrict__ aggregate_bias_imag,
    int heads,
    int dictionary_size,
    int time,
    int state,
    int chunks,
    int chunk_size) {
    const int row = blockIdx.x;
    const int chunk = blockIdx.y;
    const int lane = threadIdx.x;
    const int head = row % heads;

    extern __shared__ unsigned char shared_raw[];
    int* shared_map = reinterpret_cast<int*>(shared_raw);
    float* shared_diagonal_real = reinterpret_cast<float*>(shared_map + state);
    float* shared_diagonal_imag = shared_diagonal_real + state;
    float* shared_bias_real = shared_diagonal_imag + state;
    float* shared_bias_imag = shared_bias_real + state;
    float* shared_value_real = shared_bias_imag + state;
    float* shared_value_imag = shared_value_real + state;

    int accumulator_destination = lane;
    float accumulator_diagonal_real = 1.0f;
    float accumulator_diagonal_imag = 0.0f;
    float accumulator_bias_real = 0.0f;
    float accumulator_bias_imag = 0.0f;
    const int chunk_start = chunk * chunk_size;
    const int chunk_end = min(chunk_start + chunk_size, time);

    for (int token = chunk_start; token < chunk_end; ++token) {
        const int route = static_cast<int>(routes[row * time + token]);
        const int dictionary_offset = (head * dictionary_size + route) * state;
        const int token_offset = (row * time + token) * state + lane;
        shared_map[lane] =
            static_cast<int>(dictionary_destination[dictionary_offset + lane]);
        shared_diagonal_real[lane] = as_float(diagonal_real[token_offset]);
        shared_diagonal_imag[lane] = as_float(diagonal_imag[token_offset]);
        const float token_beta = as_float(beta[row * time + token]);
        const float previous_value_real =
            token == 0 ? 0.0f : as_float(value_real[token_offset - state]);
        const float previous_value_imag =
            token == 0 ? 0.0f : as_float(value_imag[token_offset - state]);
        complex_product(
            shared_diagonal_real[lane],
            shared_diagonal_imag[lane],
            token_beta * previous_value_real,
            token_beta * previous_value_imag,
            shared_value_real[lane],
            shared_value_imag[lane]);
        const float token_gamma = as_float(gamma[row * time + token]);
        shared_bias_real[lane] =
            token_gamma * as_float(value_real[token_offset]);
        shared_bias_imag[lane] =
            token_gamma * as_float(value_imag[token_offset]);
        __syncthreads();

        if constexpr (mode == kPermutationGather) {
            const int source =
                static_cast<int>(inverse_destination[dictionary_offset + lane]);
            shared_bias_real[lane] += shared_value_real[source];
            shared_bias_imag[lane] += shared_value_imag[source];
        } else {
            scatter_step(
                shared_map[lane],
                shared_value_real[lane],
                shared_value_imag[lane],
                shared_bias_real,
                shared_bias_imag);
        }
        __syncthreads();

        const int old_destination = accumulator_destination;
        const int next_destination = shared_map[old_destination];
        float next_diagonal_real;
        float next_diagonal_imag;
        complex_product(
            shared_diagonal_real[old_destination],
            shared_diagonal_imag[old_destination],
            accumulator_diagonal_real,
            accumulator_diagonal_imag,
            next_diagonal_real,
            next_diagonal_imag);
        complex_product(
            shared_diagonal_real[lane],
            shared_diagonal_imag[lane],
            accumulator_bias_real,
            accumulator_bias_imag,
            shared_value_real[lane],
            shared_value_imag[lane]);
        __syncthreads();

        float next_bias_real;
        float next_bias_imag;
        if constexpr (mode == kPermutationGather) {
            const int source =
                static_cast<int>(inverse_destination[dictionary_offset + lane]);
            next_bias_real = shared_bias_real[lane] + shared_value_real[source];
            next_bias_imag = shared_bias_imag[lane] + shared_value_imag[source];
        } else {
            scatter_step(
                shared_map[lane],
                shared_value_real[lane],
                shared_value_imag[lane],
                shared_bias_real,
                shared_bias_imag);
            __syncthreads();
            next_bias_real = shared_bias_real[lane];
            next_bias_imag = shared_bias_imag[lane];
        }
        __syncthreads();

        accumulator_destination = next_destination;
        accumulator_diagonal_real = next_diagonal_real;
        accumulator_diagonal_imag = next_diagonal_imag;
        accumulator_bias_real = next_bias_real;
        accumulator_bias_imag = next_bias_imag;
    }

    const int chunk_offset = (row * chunks + chunk) * state + lane;
    aggregate_destination[chunk_offset] =
        static_cast<int16_t>(accumulator_destination);
    aggregate_diagonal_real[chunk_offset] = accumulator_diagonal_real;
    aggregate_diagonal_imag[chunk_offset] = accumulator_diagonal_imag;
    aggregate_bias_real[chunk_offset] = accumulator_bias_real;
    aggregate_bias_imag[chunk_offset] = accumulator_bias_imag;
}

template <int mode>
__global__ void mamba3_phase_b_kernel(
    const int16_t* __restrict__ aggregate_destination,
    const float* __restrict__ aggregate_diagonal_real,
    const float* __restrict__ aggregate_diagonal_imag,
    float* __restrict__ aggregate_bias_real,
    float* __restrict__ aggregate_bias_imag,
    int state,
    int chunks) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;

    extern __shared__ unsigned char shared_raw[];
    int* shared_map = reinterpret_cast<int*>(shared_raw);
    float* shared_diagonal_real = reinterpret_cast<float*>(shared_map + state);
    float* shared_diagonal_imag = shared_diagonal_real + state;
    float* shared_bias_real = shared_diagonal_imag + state;
    float* shared_bias_imag = shared_bias_real + state;
    float* shared_value_real = shared_bias_imag + state;
    float* shared_value_imag = shared_value_real + state;

    int accumulator_destination = lane;
    float accumulator_diagonal_real = 1.0f;
    float accumulator_diagonal_imag = 0.0f;
    float accumulator_bias_real = 0.0f;
    float accumulator_bias_imag = 0.0f;

    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int chunk_offset = (row * chunks + chunk) * state + lane;
        shared_map[lane] =
            static_cast<int>(aggregate_destination[chunk_offset]);
        shared_diagonal_real[lane] = aggregate_diagonal_real[chunk_offset];
        shared_diagonal_imag[lane] = aggregate_diagonal_imag[chunk_offset];
        shared_bias_real[lane] = aggregate_bias_real[chunk_offset];
        shared_bias_imag[lane] = aggregate_bias_imag[chunk_offset];
        __syncthreads();

        aggregate_bias_real[chunk_offset] = accumulator_bias_real;
        aggregate_bias_imag[chunk_offset] = accumulator_bias_imag;
        const int old_destination = accumulator_destination;
        const int next_destination = shared_map[old_destination];
        const int chunk_destination = shared_map[lane];
        float next_diagonal_real;
        float next_diagonal_imag;
        complex_product(
            shared_diagonal_real[old_destination],
            shared_diagonal_imag[old_destination],
            accumulator_diagonal_real,
            accumulator_diagonal_imag,
            next_diagonal_real,
            next_diagonal_imag);
        complex_product(
            shared_diagonal_real[lane],
            shared_diagonal_imag[lane],
            accumulator_bias_real,
            accumulator_bias_imag,
            shared_value_real[lane],
            shared_value_imag[lane]);
        __syncthreads();

        float next_bias_real;
        float next_bias_imag;
        if constexpr (mode == kPermutationGather) {
            shared_map[chunk_destination] = lane;
            __syncthreads();
            permutation_step(
                lane,
                shared_map,
                shared_value_real,
                shared_value_imag,
                shared_bias_real[lane],
                shared_bias_imag[lane],
                next_bias_real,
                next_bias_imag);
        } else {
            scatter_step(
                shared_map[lane],
                shared_value_real[lane],
                shared_value_imag[lane],
                shared_bias_real,
                shared_bias_imag);
            __syncthreads();
            next_bias_real = shared_bias_real[lane];
            next_bias_imag = shared_bias_imag[lane];
        }
        __syncthreads();

        accumulator_destination = next_destination;
        accumulator_diagonal_real = next_diagonal_real;
        accumulator_diagonal_imag = next_diagonal_imag;
        accumulator_bias_real = next_bias_real;
        accumulator_bias_imag = next_bias_imag;
    }
}

template <typename scalar_t, int mode>
__global__ void mamba3_phase_c_kernel(
    const int16_t* __restrict__ dictionary_destination,
    const int16_t* __restrict__ inverse_destination,
    const int16_t* __restrict__ routes,
    const scalar_t* __restrict__ diagonal_real,
    const scalar_t* __restrict__ diagonal_imag,
    const scalar_t* __restrict__ value_real,
    const scalar_t* __restrict__ value_imag,
    const scalar_t* __restrict__ beta,
    const scalar_t* __restrict__ gamma,
    const float* __restrict__ prefix_bias_real,
    const float* __restrict__ prefix_bias_imag,
    scalar_t* __restrict__ output_real,
    scalar_t* __restrict__ output_imag,
    int heads,
    int dictionary_size,
    int time,
    int state,
    int chunks,
    int chunk_size) {
    const int row = blockIdx.x;
    const int chunk = blockIdx.y;
    const int lane = threadIdx.x;
    const int head = row % heads;

    extern __shared__ unsigned char shared_raw[];
    int* shared_map = reinterpret_cast<int*>(shared_raw);
    float* shared_diagonal_real = reinterpret_cast<float*>(shared_map + state);
    float* shared_diagonal_imag = shared_diagonal_real + state;
    float* shared_bias_real = shared_diagonal_imag + state;
    float* shared_bias_imag = shared_bias_real + state;
    float* shared_value_real = shared_bias_imag + state;
    float* shared_value_imag = shared_value_real + state;

    const int prefix_offset = (row * chunks + chunk) * state + lane;
    float state_real = prefix_bias_real[prefix_offset];
    float state_imag = prefix_bias_imag[prefix_offset];
    const int chunk_start = chunk * chunk_size;
    const int chunk_end = min(chunk_start + chunk_size, time);

    for (int token = chunk_start; token < chunk_end; ++token) {
        const int route = static_cast<int>(routes[row * time + token]);
        const int dictionary_offset = (head * dictionary_size + route) * state;
        const int token_offset = (row * time + token) * state + lane;
        shared_map[lane] =
            static_cast<int>(dictionary_destination[dictionary_offset + lane]);
        shared_diagonal_real[lane] = as_float(diagonal_real[token_offset]);
        shared_diagonal_imag[lane] = as_float(diagonal_imag[token_offset]);
        const float token_beta = as_float(beta[row * time + token]);
        const float previous_value_real =
            token == 0 ? 0.0f : as_float(value_real[token_offset - state]);
        const float previous_value_imag =
            token == 0 ? 0.0f : as_float(value_imag[token_offset - state]);
        complex_product(
            shared_diagonal_real[lane],
            shared_diagonal_imag[lane],
            token_beta * previous_value_real,
            token_beta * previous_value_imag,
            shared_value_real[lane],
            shared_value_imag[lane]);
        const float token_gamma = as_float(gamma[row * time + token]);
        shared_bias_real[lane] =
            token_gamma * as_float(value_real[token_offset]);
        shared_bias_imag[lane] =
            token_gamma * as_float(value_imag[token_offset]);
        __syncthreads();

        float token_bias_real;
        float token_bias_imag;
        if constexpr (mode == kPermutationGather) {
            const int source =
                static_cast<int>(inverse_destination[dictionary_offset + lane]);
            token_bias_real = shared_bias_real[lane] + shared_value_real[source];
            token_bias_imag = shared_bias_imag[lane] + shared_value_imag[source];
        } else {
            scatter_step(
                shared_map[lane],
                shared_value_real[lane],
                shared_value_imag[lane],
                shared_bias_real,
                shared_bias_imag);
            __syncthreads();
            token_bias_real = shared_bias_real[lane];
            token_bias_imag = shared_bias_imag[lane];
        }
        if constexpr (mode == kPermutationGather) {
            __syncthreads();
        }

        complex_product(
            shared_diagonal_real[lane],
            shared_diagonal_imag[lane],
            state_real,
            state_imag,
            shared_value_real[lane],
            shared_value_imag[lane]);
        if constexpr (mode == kGeneralScatter) {
            shared_bias_real[lane] = token_bias_real;
            shared_bias_imag[lane] = token_bias_imag;
        }
        __syncthreads();

        float next_real;
        float next_imag;
        if constexpr (mode == kPermutationGather) {
            const int source =
                static_cast<int>(inverse_destination[dictionary_offset + lane]);
            next_real = token_bias_real + shared_value_real[source];
            next_imag = token_bias_imag + shared_value_imag[source];
        } else {
            scatter_step(
                shared_map[lane],
                shared_value_real[lane],
                shared_value_imag[lane],
                shared_bias_real,
                shared_bias_imag);
            __syncthreads();
            next_real = shared_bias_real[lane];
            next_imag = shared_bias_imag[lane];
        }
        __syncthreads();

        state_real = next_real;
        state_imag = next_imag;
        output_real[token_offset] = static_cast<scalar_t>(state_real);
        output_imag[token_offset] = static_cast<scalar_t>(state_imag);
    }
}

template <typename scalar_t>
__global__ void backward_kernel(
    const int16_t* __restrict__ dictionary_destination,
    const int16_t* __restrict__ routes,
    const scalar_t* __restrict__ diagonal_real,
    const scalar_t* __restrict__ diagonal_imag,
    const scalar_t* __restrict__ output_real,
    const scalar_t* __restrict__ output_imag,
    const scalar_t* __restrict__ grad_output_real,
    const scalar_t* __restrict__ grad_output_imag,
    scalar_t* __restrict__ grad_diagonal_real,
    scalar_t* __restrict__ grad_diagonal_imag,
    scalar_t* __restrict__ grad_bias_real,
    scalar_t* __restrict__ grad_bias_imag,
    int heads,
    int dictionary_size,
    int time,
    int state) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    if (lane >= state) {
        return;
    }
    const int head = row % heads;
    extern __shared__ float shared_gradient[];
    float* shared_gradient_real = shared_gradient;
    float* shared_gradient_imag = shared_gradient_real + state;

    float carry_real = 0.0f;
    float carry_imag = 0.0f;
    for (int token = time - 1; token >= 0; --token) {
        const int token_offset = (row * time + token) * state + lane;
        const float total_real =
            as_float(grad_output_real[token_offset]) + carry_real;
        const float total_imag =
            as_float(grad_output_imag[token_offset]) + carry_imag;
        shared_gradient_real[lane] = total_real;
        shared_gradient_imag[lane] = total_imag;
        grad_bias_real[token_offset] = static_cast<scalar_t>(total_real);
        grad_bias_imag[token_offset] = static_cast<scalar_t>(total_imag);
        __syncthreads();

        const int route = static_cast<int>(routes[row * time + token]);
        const int dictionary_offset =
            (head * dictionary_size + route) * state;
        const int destination =
            static_cast<int>(dictionary_destination[dictionary_offset + lane]);
        const float destination_gradient_real =
            shared_gradient_real[destination];
        const float destination_gradient_imag =
            shared_gradient_imag[destination];
        const float previous_real =
            token == 0 ? 0.0f : as_float(output_real[token_offset - state]);
        const float previous_imag =
            token == 0 ? 0.0f : as_float(output_imag[token_offset - state]);
        const float token_diagonal_real = as_float(diagonal_real[token_offset]);
        const float token_diagonal_imag = as_float(diagonal_imag[token_offset]);

        grad_diagonal_real[token_offset] = static_cast<scalar_t>(
            destination_gradient_real * previous_real +
            destination_gradient_imag * previous_imag);
        grad_diagonal_imag[token_offset] = static_cast<scalar_t>(
            -destination_gradient_real * previous_imag +
            destination_gradient_imag * previous_real);
        carry_real =
            destination_gradient_real * token_diagonal_real +
            destination_gradient_imag * token_diagonal_imag;
        carry_imag =
            -destination_gradient_real * token_diagonal_imag +
            destination_gradient_imag * token_diagonal_real;
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void paper_backward_phase_a_kernel(
    const int16_t* __restrict__ dictionary_destination,
    const int16_t* __restrict__ routes,
    const scalar_t* __restrict__ diagonal_real,
    const scalar_t* __restrict__ diagonal_imag,
    const scalar_t* __restrict__ grad_output_real,
    const scalar_t* __restrict__ grad_output_imag,
    int16_t* __restrict__ aggregate_destination,
    float* __restrict__ aggregate_diagonal_real,
    float* __restrict__ aggregate_diagonal_imag,
    float* __restrict__ aggregate_bias_real,
    float* __restrict__ aggregate_bias_imag,
    int heads,
    int dictionary_size,
    int time,
    int state,
    int chunks,
    int chunk_size) {
    const int row = blockIdx.x;
    const int chunk = blockIdx.y;
    const int lane = threadIdx.x;
    const int head = row % heads;

    extern __shared__ unsigned char shared_raw[];
    int* shared_map = reinterpret_cast<int*>(shared_raw);
    float* shared_diagonal_real = reinterpret_cast<float*>(shared_map + state);
    float* shared_diagonal_imag = shared_diagonal_real + state;
    float* shared_bias_real = shared_diagonal_imag + state;
    float* shared_bias_imag = shared_bias_real + state;
    float* shared_gradient_real = shared_bias_imag + state;
    float* shared_gradient_imag = shared_gradient_real + state;

    int accumulator_destination = lane;
    float accumulator_diagonal_real = 1.0f;
    float accumulator_diagonal_imag = 0.0f;
    float accumulator_bias_real = 0.0f;
    float accumulator_bias_imag = 0.0f;
    const int chunk_start = chunk * chunk_size;
    const int chunk_end = min(chunk_start + chunk_size, time);

    for (int token = chunk_end - 1; token >= chunk_start; --token) {
        const int token_offset = (row * time + token) * state + lane;
        shared_map[lane] = accumulator_destination;
        shared_diagonal_real[lane] = accumulator_diagonal_real;
        shared_diagonal_imag[lane] = accumulator_diagonal_imag;
        shared_bias_real[lane] = accumulator_bias_real;
        shared_bias_imag[lane] = accumulator_bias_imag;
        shared_gradient_real[lane] = as_float(grad_output_real[token_offset]);
        shared_gradient_imag[lane] = as_float(grad_output_imag[token_offset]);
        __syncthreads();

        const int route = static_cast<int>(routes[row * time + token]);
        const int dictionary_offset = (head * dictionary_size + route) * state;
        const int destination =
            static_cast<int>(dictionary_destination[dictionary_offset + lane]);
        const float token_diagonal_real = as_float(diagonal_real[token_offset]);
        const float token_diagonal_imag = as_float(diagonal_imag[token_offset]);
        const int next_destination = shared_map[destination];
        float next_diagonal_real;
        float next_diagonal_imag;
        complex_product(
            token_diagonal_real,
            -token_diagonal_imag,
            shared_diagonal_real[destination],
            shared_diagonal_imag[destination],
            next_diagonal_real,
            next_diagonal_imag);
        const float affine_real =
            shared_gradient_real[destination] + shared_bias_real[destination];
        const float affine_imag =
            shared_gradient_imag[destination] + shared_bias_imag[destination];
        float next_bias_real;
        float next_bias_imag;
        complex_product(
            token_diagonal_real,
            -token_diagonal_imag,
            affine_real,
            affine_imag,
            next_bias_real,
            next_bias_imag);
        __syncthreads();

        accumulator_destination = next_destination;
        accumulator_diagonal_real = next_diagonal_real;
        accumulator_diagonal_imag = next_diagonal_imag;
        accumulator_bias_real = next_bias_real;
        accumulator_bias_imag = next_bias_imag;
    }

    const int chunk_offset = (row * chunks + chunk) * state + lane;
    aggregate_destination[chunk_offset] =
        static_cast<int16_t>(accumulator_destination);
    aggregate_diagonal_real[chunk_offset] = accumulator_diagonal_real;
    aggregate_diagonal_imag[chunk_offset] = accumulator_diagonal_imag;
    aggregate_bias_real[chunk_offset] = accumulator_bias_real;
    aggregate_bias_imag[chunk_offset] = accumulator_bias_imag;
}

__global__ void paper_backward_phase_b_kernel(
    const int16_t* __restrict__ aggregate_destination,
    const float* __restrict__ aggregate_diagonal_real,
    const float* __restrict__ aggregate_diagonal_imag,
    const float* __restrict__ aggregate_bias_real,
    const float* __restrict__ aggregate_bias_imag,
    float* __restrict__ chunk_carry_real,
    float* __restrict__ chunk_carry_imag,
    int state,
    int chunks) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    extern __shared__ float shared_carry[];
    float* shared_carry_real = shared_carry;
    float* shared_carry_imag = shared_carry_real + state;
    float carry_real = 0.0f;
    float carry_imag = 0.0f;

    for (int chunk = chunks - 1; chunk >= 0; --chunk) {
        const int chunk_offset = (row * chunks + chunk) * state + lane;
        chunk_carry_real[chunk_offset] = carry_real;
        chunk_carry_imag[chunk_offset] = carry_imag;
        shared_carry_real[lane] = carry_real;
        shared_carry_imag[lane] = carry_imag;
        __syncthreads();

        const int destination =
            static_cast<int>(aggregate_destination[chunk_offset]);
        float transformed_real;
        float transformed_imag;
        complex_product(
            aggregate_diagonal_real[chunk_offset],
            aggregate_diagonal_imag[chunk_offset],
            shared_carry_real[destination],
            shared_carry_imag[destination],
            transformed_real,
            transformed_imag);
        carry_real = transformed_real + aggregate_bias_real[chunk_offset];
        carry_imag = transformed_imag + aggregate_bias_imag[chunk_offset];
        __syncthreads();
    }
}

__device__ __forceinline__ float block_sum(
    float value,
    float* warp_sums,
    int state) {
    const int lane = threadIdx.x;
    const int warp_lane = lane & 31;
    const int warp = lane >> 5;
    const int warp_start = warp << 5;
    const int warp_width = min(32, state - warp_start);
    const unsigned active = __activemask();
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_down_sync(active, value, offset);
        if (warp_lane + offset < warp_width) {
            value += other;
        }
    }
    if (warp_lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        const int warps = (state + 31) / 32;
        float total = warp_lane < warps ? warp_sums[warp_lane] : 0.0f;
        const unsigned first_warp = __activemask();
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other = __shfl_down_sync(first_warp, total, offset);
            if (warp_lane + offset < warps) {
                total += other;
            }
        }
        if (warp_lane == 0) {
            warp_sums[0] = total;
        }
    }
    __syncthreads();
    return warp_sums[0];
}

__device__ __forceinline__ float state_sum(
    float value,
    float* warp_sums,
    int state) {
    const int lane = threadIdx.x;
    const int warp_lane = lane & 31;
    const int warp = lane >> 5;
    const int warp_start = warp << 5;
    const int warp_width = min(32, state - warp_start);
    const unsigned active = __activemask();
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_down_sync(active, value, offset);
        if (warp_lane + offset < warp_width) {
            value += other;
        }
    }
    if (state <= 32) {
        return __shfl_sync(active, value, 0);
    }
    if (warp_lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        const int warps = (state + 31) / 32;
        float total = warp_lane < warps ? warp_sums[warp_lane] : 0.0f;
        const unsigned first_warp = __activemask();
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other = __shfl_down_sync(first_warp, total, offset);
            if (warp_lane + offset < warps) {
                total += other;
            }
        }
        if (warp_lane == 0) {
            warp_sums[0] = total;
        }
    }
    __syncthreads();
    return warp_sums[0];
}

template <typename scalar_t>
__global__ void mamba3_backward_fused_kernel(
    const float* __restrict__ selector_logits,
    const int16_t* __restrict__ dictionary_destination,
    const int16_t* __restrict__ routes,
    const scalar_t* __restrict__ diagonal_real,
    const scalar_t* __restrict__ diagonal_imag,
    const scalar_t* __restrict__ output_real,
    const scalar_t* __restrict__ output_imag,
    const scalar_t* __restrict__ grad_output_real,
    const scalar_t* __restrict__ grad_output_imag,
    const scalar_t* __restrict__ value_real,
    const scalar_t* __restrict__ value_imag,
    const scalar_t* __restrict__ beta,
    const scalar_t* __restrict__ gamma,
    scalar_t* __restrict__ grad_diagonal_real,
    scalar_t* __restrict__ grad_diagonal_imag,
    scalar_t* __restrict__ grad_value_real,
    scalar_t* __restrict__ grad_value_imag,
    scalar_t* __restrict__ grad_beta,
    scalar_t* __restrict__ grad_gamma,
    float* __restrict__ active_dictionary_gradient,
    float* __restrict__ grad_selector_logits,
    int heads,
    int dictionary_size,
    int time,
    int state,
    bool aggregate_dictionary,
    float router_inverse_temperature) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    const int batch = row / heads;
    const int head = row % heads;
    extern __shared__ float shared[];
    float* shared_gradient_real = shared;
    float* shared_gradient_imag = shared_gradient_real + state;
    float* shared_active_dictionary = shared_gradient_imag + state;
    const int local_dictionary_elements =
        aggregate_dictionary ? dictionary_size * state : 0;
    float* shared_warp_sums =
        shared_active_dictionary + local_dictionary_elements;
    float* shared_router = shared_warp_sums + 32;
    for (int index = lane; index < local_dictionary_elements; index += state) {
        shared_active_dictionary[index] = 0.0f;
    }
    __syncthreads();

    float carry_real = 0.0f;
    float carry_imag = 0.0f;
    for (int token = time - 1; token >= 0; --token) {
        const int token_offset = (row * time + token) * state + lane;
        const float total_real =
            as_float(grad_output_real[token_offset]) + carry_real;
        const float total_imag =
            as_float(grad_output_imag[token_offset]) + carry_imag;
        shared_gradient_real[lane] = total_real;
        shared_gradient_imag[lane] = total_imag;
        __syncthreads();

        const int route = static_cast<int>(routes[row * time + token]);
        const int dictionary_offset = (head * dictionary_size + route) * state;
        const int destination =
            static_cast<int>(dictionary_destination[dictionary_offset + lane]);
        const float destination_gradient_real =
            shared_gradient_real[destination];
        const float destination_gradient_imag =
            shared_gradient_imag[destination];
        const float previous_real =
            token == 0 ? 0.0f : as_float(output_real[token_offset - state]);
        const float previous_imag =
            token == 0 ? 0.0f : as_float(output_imag[token_offset - state]);
        const float previous_value_real =
            token == 0 ? 0.0f : as_float(value_real[token_offset - state]);
        const float previous_value_imag =
            token == 0 ? 0.0f : as_float(value_imag[token_offset - state]);
        const float token_beta = as_float(beta[row * time + token]);
        const float transition_input_real =
            previous_real + token_beta * previous_value_real;
        const float transition_input_imag =
            previous_imag + token_beta * previous_value_imag;
        const float token_diagonal_real = as_float(diagonal_real[token_offset]);
        const float token_diagonal_imag = as_float(diagonal_imag[token_offset]);
        float transformed_input_real;
        float transformed_input_imag;
        complex_product(
            token_diagonal_real,
            token_diagonal_imag,
            transition_input_real,
            transition_input_imag,
            transformed_input_real,
            transformed_input_imag);
        const float active =
            destination_gradient_real * transformed_input_real +
            destination_gradient_imag * transformed_input_imag;
        if (aggregate_dictionary) {
            shared_active_dictionary[route * state + lane] += active;
        } else {
            atomicAdd(
                active_dictionary_gradient + dictionary_offset + lane,
                active);
        }

        const float selector_score =
            state_sum(active, shared_warp_sums, state);
        const int logits_offset =
            ((batch * time + token) * heads + head) * dictionary_size;
        if (lane == 0) {
            float maximum = -INFINITY;
            for (int dictionary = 0; dictionary < dictionary_size; ++dictionary) {
                maximum = fmaxf(
                    maximum,
                    selector_logits[logits_offset + dictionary] *
                        router_inverse_temperature);
            }
            float denominator = 0.0f;
            for (int dictionary = 0; dictionary < dictionary_size; ++dictionary) {
                denominator += expf(
                    selector_logits[logits_offset + dictionary] *
                            router_inverse_temperature -
                        maximum);
            }
            shared_router[0] = maximum;
            shared_router[1] = denominator;
            shared_router[2] = expf(
                                   selector_logits[logits_offset + route] *
                                           router_inverse_temperature -
                                       maximum) /
                denominator;
            shared_router[3] = selector_score;
            shared_router[4] = static_cast<float>(route);
        }
        __syncthreads();
        for (int dictionary = lane;
             dictionary < dictionary_size;
             dictionary += state) {
            const float probability =
                expf(selector_logits[logits_offset + dictionary] *
                             router_inverse_temperature -
                     shared_router[0]) /
                shared_router[1];
            const float indicator =
                dictionary == static_cast<int>(shared_router[4]) ? 1.0f : 0.0f;
            grad_selector_logits[logits_offset + dictionary] =
                shared_router[3] * shared_router[2] *
                (indicator - probability) * router_inverse_temperature;
        }

        grad_diagonal_real[token_offset] = static_cast<scalar_t>(
            destination_gradient_real * transition_input_real +
            destination_gradient_imag * transition_input_imag);
        grad_diagonal_imag[token_offset] = static_cast<scalar_t>(
            -destination_gradient_real * transition_input_imag +
            destination_gradient_imag * transition_input_real);
        const float future_carry_real = carry_real;
        const float future_carry_imag = carry_imag;
        carry_real =
            destination_gradient_real * token_diagonal_real +
            destination_gradient_imag * token_diagonal_imag;
        carry_imag =
            -destination_gradient_real * token_diagonal_imag +
            destination_gradient_imag * token_diagonal_real;
        const float token_gamma = as_float(gamma[row * time + token]);
        const float next_beta =
            token + 1 < time ? as_float(beta[row * time + token + 1]) : 0.0f;
        grad_value_real[token_offset] = static_cast<scalar_t>(
            token_gamma * total_real + next_beta * future_carry_real);
        grad_value_imag[token_offset] = static_cast<scalar_t>(
            token_gamma * total_imag + next_beta * future_carry_imag);
        const float beta_component =
            carry_real * previous_value_real +
            carry_imag * previous_value_imag;
        const float gamma_component =
            total_real * as_float(value_real[token_offset]) +
            total_imag * as_float(value_imag[token_offset]);
        const float beta_gradient =
            state_sum(beta_component, shared_warp_sums, state);
        const float gamma_gradient =
            state_sum(gamma_component, shared_warp_sums, state);
        if (lane == 0) {
            grad_beta[row * time + token] =
                static_cast<scalar_t>(beta_gradient);
            grad_gamma[row * time + token] =
                static_cast<scalar_t>(gamma_gradient);
        }
        __syncthreads();
    }

    if (aggregate_dictionary) {
        for (int dictionary = 0; dictionary < dictionary_size; ++dictionary) {
            const float active =
                shared_active_dictionary[dictionary * state + lane];
            if (active != 0.0f) {
                atomicAdd(
                    active_dictionary_gradient +
                        (head * dictionary_size + dictionary) * state + lane,
                    active);
            }
        }
    }
}

template <typename scalar_t>
__global__ void paper_backward_phase_c_kernel(
    const int16_t* __restrict__ dictionary_destination,
    const int16_t* __restrict__ routes,
    const scalar_t* __restrict__ diagonal_real,
    const scalar_t* __restrict__ diagonal_imag,
    const scalar_t* __restrict__ bias_real,
    const scalar_t* __restrict__ bias_imag,
    const scalar_t* __restrict__ output_real,
    const scalar_t* __restrict__ output_imag,
    const scalar_t* __restrict__ grad_output_real,
    const scalar_t* __restrict__ grad_output_imag,
    const scalar_t* __restrict__ value_real,
    const scalar_t* __restrict__ value_imag,
    const scalar_t* __restrict__ beta,
    const scalar_t* __restrict__ gamma,
    const float* __restrict__ chunk_carry_real,
    const float* __restrict__ chunk_carry_imag,
    scalar_t* __restrict__ grad_diagonal_real,
    scalar_t* __restrict__ grad_diagonal_imag,
    scalar_t* __restrict__ grad_bias_real,
    scalar_t* __restrict__ grad_bias_imag,
    scalar_t* __restrict__ grad_beta,
    scalar_t* __restrict__ grad_gamma,
    float* __restrict__ active_dictionary_gradient,
    float* __restrict__ selector_score,
    int heads,
    int dictionary_size,
    int time,
    int state,
    int chunks,
    int chunk_size,
    bool aggregate_dictionary,
    bool mamba3_siso) {
    const int row = blockIdx.x;
    const int chunk = blockIdx.y;
    const int lane = threadIdx.x;
    const int head = row % heads;
    extern __shared__ float shared[];
    float* shared_gradient_real = shared;
    float* shared_gradient_imag = shared_gradient_real + state;
    float* shared_active_dictionary = shared_gradient_imag + state;
    const int local_dictionary_elements =
        aggregate_dictionary ? dictionary_size * state : 0;
    float* shared_warp_sums =
        shared_active_dictionary + local_dictionary_elements;
    for (int index = lane; index < local_dictionary_elements; index += state) {
        shared_active_dictionary[index] = 0.0f;
    }
    __syncthreads();

    const int chunk_offset = (row * chunks + chunk) * state + lane;
    float carry_real = chunk_carry_real[chunk_offset];
    float carry_imag = chunk_carry_imag[chunk_offset];
    const int chunk_start = chunk * chunk_size;
    const int chunk_end = min(chunk_start + chunk_size, time);

    for (int token = chunk_end - 1; token >= chunk_start; --token) {
        const int token_offset = (row * time + token) * state + lane;
        const float total_real =
            as_float(grad_output_real[token_offset]) + carry_real;
        const float total_imag =
            as_float(grad_output_imag[token_offset]) + carry_imag;
        shared_gradient_real[lane] = total_real;
        shared_gradient_imag[lane] = total_imag;
        if (!mamba3_siso) {
            grad_bias_real[token_offset] = static_cast<scalar_t>(total_real);
            grad_bias_imag[token_offset] = static_cast<scalar_t>(total_imag);
        }
        __syncthreads();

        const int route = static_cast<int>(routes[row * time + token]);
        const int dictionary_offset = (head * dictionary_size + route) * state;
        const int destination =
            static_cast<int>(dictionary_destination[dictionary_offset + lane]);
        const float destination_gradient_real =
            shared_gradient_real[destination];
        const float destination_gradient_imag =
            shared_gradient_imag[destination];
        const float previous_real =
            token == 0 ? 0.0f : as_float(output_real[token_offset - state]);
        const float previous_imag =
            token == 0 ? 0.0f : as_float(output_imag[token_offset - state]);
        const float previous_value_real =
            !mamba3_siso || token == 0
                ? 0.0f
                : as_float(value_real[token_offset - state]);
        const float previous_value_imag =
            !mamba3_siso || token == 0
                ? 0.0f
                : as_float(value_imag[token_offset - state]);
        const float token_beta =
            mamba3_siso ? as_float(beta[row * time + token]) : 0.0f;
        const float transition_input_real =
            previous_real + token_beta * previous_value_real;
        const float transition_input_imag =
            previous_imag + token_beta * previous_value_imag;
        const float token_diagonal_real = as_float(diagonal_real[token_offset]);
        const float token_diagonal_imag = as_float(diagonal_imag[token_offset]);
        float transformed_input_real;
        float transformed_input_imag;
        complex_product(
            token_diagonal_real,
            token_diagonal_imag,
            transition_input_real,
            transition_input_imag,
            transformed_input_real,
            transformed_input_imag);
        const float active =
            destination_gradient_real * transformed_input_real +
            destination_gradient_imag * transformed_input_imag;
        if (aggregate_dictionary) {
            shared_active_dictionary[route * state + lane] += active;
        } else {
            atomicAdd(
                active_dictionary_gradient + dictionary_offset + lane,
                active);
        }
        const float score = block_sum(
            active,
            shared_warp_sums,
            state);
        if (lane == 0) {
            selector_score[row * time + token] = score;
        }
        grad_diagonal_real[token_offset] = static_cast<scalar_t>(
            destination_gradient_real * transition_input_real +
            destination_gradient_imag * transition_input_imag);
        grad_diagonal_imag[token_offset] = static_cast<scalar_t>(
            -destination_gradient_real * transition_input_imag +
            destination_gradient_imag * transition_input_real);
        const float future_carry_real = carry_real;
        const float future_carry_imag = carry_imag;
        carry_real =
            destination_gradient_real * token_diagonal_real +
            destination_gradient_imag * token_diagonal_imag;
        carry_imag =
            -destination_gradient_real * token_diagonal_imag +
            destination_gradient_imag * token_diagonal_real;
        if (mamba3_siso) {
            const float token_gamma = as_float(gamma[row * time + token]);
            const float next_beta =
                token + 1 < time ? as_float(beta[row * time + token + 1]) : 0.0f;
            grad_bias_real[token_offset] = static_cast<scalar_t>(
                token_gamma * total_real + next_beta * future_carry_real);
            grad_bias_imag[token_offset] = static_cast<scalar_t>(
                token_gamma * total_imag + next_beta * future_carry_imag);
            const float beta_component =
                carry_real * previous_value_real +
                carry_imag * previous_value_imag;
            const float gamma_component =
                total_real * as_float(value_real[token_offset]) +
                total_imag * as_float(value_imag[token_offset]);
            const float beta_gradient = block_sum(
                beta_component,
                shared_warp_sums,
                state);
            const float gamma_gradient = block_sum(
                gamma_component,
                shared_warp_sums,
                state);
            if (lane == 0) {
                grad_beta[row * time + token] =
                    static_cast<scalar_t>(beta_gradient);
                grad_gamma[row * time + token] =
                    static_cast<scalar_t>(gamma_gradient);
            }
        }
        __syncthreads();
    }

    if (aggregate_dictionary) {
        for (int dictionary = 0; dictionary < dictionary_size; ++dictionary) {
            const float active =
                shared_active_dictionary[dictionary * state + lane];
            if (active != 0.0f) {
                atomicAdd(
                    active_dictionary_gradient +
                        (head * dictionary_size + dictionary) * state + lane,
                    active);
            }
        }
    }
}

__global__ void paper_dictionary_gradient_kernel(
    const float* __restrict__ dictionary_logits,
    const int16_t* __restrict__ dictionary_destination,
    const float* __restrict__ active_gradient,
    float* __restrict__ gradient,
    int dictionary_size,
    int state,
    float inverse_temperature) {
    const int head = blockIdx.x;
    const int dictionary = blockIdx.y;
    const int source = blockIdx.z;
    const int destination = threadIdx.x;
    const int column_offset =
        ((head * dictionary_size + dictionary) * state * state) + source;
    extern __shared__ float shared[];
    if (destination == 0) {
        float maximum = -INFINITY;
        for (int row = 0; row < state; ++row) {
            maximum = fmaxf(
                maximum,
                dictionary_logits[column_offset + row * state] *
                    inverse_temperature);
        }
        float denominator = 0.0f;
        for (int row = 0; row < state; ++row) {
            denominator += expf(
                dictionary_logits[column_offset + row * state] *
                        inverse_temperature -
                    maximum);
        }
        const int selected = static_cast<int>(
            dictionary_destination[
                (head * dictionary_size + dictionary) * state + source]);
        shared[0] = maximum;
        shared[1] = denominator;
        shared[2] = expf(
                        dictionary_logits[column_offset + selected * state] *
                            inverse_temperature -
                        maximum) /
                    denominator;
        shared[3] =
            active_gradient[
                (head * dictionary_size + dictionary) * state + source];
        shared[4] = static_cast<float>(selected);
    }
    __syncthreads();
    const float probability =
        expf(dictionary_logits[column_offset + destination * state] *
                     inverse_temperature -
             shared[0]) /
        shared[1];
    const float indicator =
        destination == static_cast<int>(shared[4]) ? 1.0f : 0.0f;
    gradient[column_offset + destination * state] =
        shared[3] * shared[2] * (indicator - probability) *
        inverse_temperature;
}

__global__ void paper_selector_gradient_kernel(
    const float* __restrict__ selector_logits,
    const int16_t* __restrict__ routes,
    const float* __restrict__ selector_score,
    float* __restrict__ gradient,
    int heads,
    int dictionary_size,
    int time,
    float inverse_temperature) {
    const int row = blockIdx.x;
    const int token = blockIdx.y;
    const int dictionary = threadIdx.x;
    const int batch = row / heads;
    const int head = row % heads;
    const int logits_offset =
        ((batch * time + token) * heads + head) * dictionary_size;
    extern __shared__ float shared[];
    if (dictionary == 0) {
        float maximum = -INFINITY;
        for (int k = 0; k < dictionary_size; ++k) {
            maximum = fmaxf(
                maximum,
                selector_logits[logits_offset + k] * inverse_temperature);
        }
        float denominator = 0.0f;
        for (int k = 0; k < dictionary_size; ++k) {
            denominator += expf(
                selector_logits[logits_offset + k] * inverse_temperature -
                maximum);
        }
        const int selected = static_cast<int>(routes[row * time + token]);
        shared[0] = maximum;
        shared[1] = denominator;
        shared[2] = expf(
                        selector_logits[logits_offset + selected] *
                            inverse_temperature -
                        maximum) /
                    denominator;
        shared[3] = selector_score[row * time + token];
        shared[4] = static_cast<float>(selected);
    }
    __syncthreads();
    const float probability =
        expf(selector_logits[logits_offset + dictionary] *
                     inverse_temperature -
             shared[0]) /
        shared[1];
    const float indicator =
        dictionary == static_cast<int>(shared[4]) ? 1.0f : 0.0f;
    gradient[logits_offset + dictionary] =
        shared[3] * shared[2] * (indicator - probability) *
        inverse_temperature;
}

void validate_forward_inputs(
    const torch::Tensor& destination,
    const torch::Tensor& inverse_destination,
    const torch::Tensor& routes,
    const torch::Tensor& diagonal_real,
    const torch::Tensor& diagonal_imag,
    const torch::Tensor& bias_real,
    const torch::Tensor& bias_imag,
    int64_t chunk_size,
    int64_t mode) {
    TORCH_CHECK(destination.is_cuda(), "destination must be CUDA");
    TORCH_CHECK(destination.scalar_type() == torch::kInt16, "destination must be int16");
    TORCH_CHECK(inverse_destination.scalar_type() == torch::kInt16, "inverse must be int16");
    TORCH_CHECK(routes.scalar_type() == torch::kInt16, "routes must be int16");
    TORCH_CHECK(destination.dim() == 3, "destination must be (H,K,N)");
    TORCH_CHECK(routes.dim() == 3, "routes must be (B,H,T)");
    TORCH_CHECK(diagonal_real.dim() == 4, "values must be (B,H,T,N)");
    TORCH_CHECK(
        diagonal_real.sizes() == diagonal_imag.sizes() &&
            diagonal_real.sizes() == bias_real.sizes() &&
            diagonal_real.sizes() == bias_imag.sizes(),
        "split value shapes must match");
    TORCH_CHECK(
        diagonal_real.scalar_type() == torch::kFloat32 ||
            diagonal_real.scalar_type() == torch::kBFloat16,
        "native kernel supports float32 and bfloat16");
    TORCH_CHECK(
        diagonal_real.scalar_type() == diagonal_imag.scalar_type() &&
            diagonal_real.scalar_type() == bias_real.scalar_type() &&
            diagonal_real.scalar_type() == bias_imag.scalar_type(),
        "split value dtypes must match");
    TORCH_CHECK(diagonal_real.size(3) < 1024, "state must be below 1024");
    TORCH_CHECK(diagonal_real.size(3) > 0, "state must be positive");
    TORCH_CHECK(diagonal_real.size(2) > 0, "time must be positive");
    TORCH_CHECK(chunk_size > 0 && chunk_size <= 128, "chunk_size must be in [1,128]");
    TORCH_CHECK(
        mode == kGeneralScatter || mode == kPermutationGather,
        "invalid transition mode");
}

}  // namespace

std::vector<torch::Tensor> flash_pd_native_forward_cuda(
    torch::Tensor destination,
    torch::Tensor inverse_destination,
    torch::Tensor routes,
    torch::Tensor diagonal_real,
    torch::Tensor diagonal_imag,
    torch::Tensor bias_real,
    torch::Tensor bias_imag,
    int64_t chunk_size,
    int64_t mode) {
    validate_forward_inputs(
        destination,
        inverse_destination,
        routes,
        diagonal_real,
        diagonal_imag,
        bias_real,
        bias_imag,
        chunk_size,
        mode);
    c10::cuda::CUDAGuard device_guard(diagonal_real.device());
    const int batch = diagonal_real.size(0);
    const int heads = diagonal_real.size(1);
    const int time = diagonal_real.size(2);
    const int state = diagonal_real.size(3);
    const int dictionary_size = destination.size(1);
    const int rows = batch * heads;
    const int chunks = (time + chunk_size - 1) / chunk_size;
    const size_t shared_bytes = static_cast<size_t>(28) * state;

    auto destination_options = destination.options().dtype(torch::kInt16);
    auto float_options = diagonal_real.options().dtype(torch::kFloat32);
    auto aggregate_destination =
        torch::empty({rows, chunks, state}, destination_options);
    auto aggregate_diagonal_real =
        torch::empty({rows, chunks, state}, float_options);
    auto aggregate_diagonal_imag = torch::empty_like(aggregate_diagonal_real);
    auto aggregate_bias_real = torch::empty_like(aggregate_diagonal_real);
    auto aggregate_bias_imag = torch::empty_like(aggregate_diagonal_real);
    auto prefix_destination = torch::empty_like(aggregate_destination);
    auto prefix_diagonal_real = torch::empty_like(aggregate_diagonal_real);
    auto prefix_diagonal_imag = torch::empty_like(aggregate_diagonal_real);
    auto prefix_bias_real = torch::empty_like(aggregate_diagonal_real);
    auto prefix_bias_imag = torch::empty_like(aggregate_diagonal_real);
    auto output_real = torch::empty_like(diagonal_real);
    auto output_imag = torch::empty_like(diagonal_imag);
    const auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND(
        at::ScalarType::BFloat16,
        diagonal_real.scalar_type(),
        "flash_pd_native_forward",
        [&] {
            phase_a_kernel<scalar_t><<<
                dim3(rows, chunks),
                state,
                shared_bytes,
                stream>>>(
                destination.data_ptr<int16_t>(),
                routes.data_ptr<int16_t>(),
                diagonal_real.data_ptr<scalar_t>(),
                diagonal_imag.data_ptr<scalar_t>(),
                bias_real.data_ptr<scalar_t>(),
                bias_imag.data_ptr<scalar_t>(),
                aggregate_destination.data_ptr<int16_t>(),
                aggregate_diagonal_real.data_ptr<float>(),
                aggregate_diagonal_imag.data_ptr<float>(),
                aggregate_bias_real.data_ptr<float>(),
                aggregate_bias_imag.data_ptr<float>(),
                heads,
                dictionary_size,
                time,
                state,
                chunks,
                chunk_size,
                mode);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    phase_b_kernel<<<rows, state, shared_bytes, stream>>>(
        aggregate_destination.data_ptr<int16_t>(),
        aggregate_diagonal_real.data_ptr<float>(),
        aggregate_diagonal_imag.data_ptr<float>(),
        aggregate_bias_real.data_ptr<float>(),
        aggregate_bias_imag.data_ptr<float>(),
        prefix_destination.data_ptr<int16_t>(),
        prefix_diagonal_real.data_ptr<float>(),
        prefix_diagonal_imag.data_ptr<float>(),
        prefix_bias_real.data_ptr<float>(),
        prefix_bias_imag.data_ptr<float>(),
        state,
        chunks,
        mode);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    AT_DISPATCH_FLOATING_TYPES_AND(
        at::ScalarType::BFloat16,
        diagonal_real.scalar_type(),
        "flash_pd_native_replay",
        [&] {
            phase_c_kernel<scalar_t><<<
                dim3(rows, chunks),
                state,
                shared_bytes,
                stream>>>(
                destination.data_ptr<int16_t>(),
                routes.data_ptr<int16_t>(),
                diagonal_real.data_ptr<scalar_t>(),
                diagonal_imag.data_ptr<scalar_t>(),
                bias_real.data_ptr<scalar_t>(),
                bias_imag.data_ptr<scalar_t>(),
                prefix_bias_real.data_ptr<float>(),
                prefix_bias_imag.data_ptr<float>(),
                output_real.data_ptr<scalar_t>(),
                output_imag.data_ptr<scalar_t>(),
                heads,
                dictionary_size,
                time,
                state,
                chunks,
                chunk_size,
                mode);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output_real, output_imag};
}

std::vector<torch::Tensor> flash_pd_native_mamba3_forward_cuda(
    torch::Tensor destination,
    torch::Tensor inverse_destination,
    torch::Tensor routes,
    torch::Tensor diagonal_real,
    torch::Tensor diagonal_imag,
    torch::Tensor value_real,
    torch::Tensor value_imag,
    torch::Tensor beta,
    torch::Tensor gamma,
    int64_t chunk_size,
    int64_t mode) {
    validate_forward_inputs(
        destination,
        inverse_destination,
        routes,
        diagonal_real,
        diagonal_imag,
        value_real,
        value_imag,
        chunk_size,
        mode);
    TORCH_CHECK(
        beta.sizes() ==
                torch::IntArrayRef(
                    {diagonal_real.size(0), diagonal_real.size(1), diagonal_real.size(2)}) &&
            gamma.sizes() == beta.sizes(),
        "beta and gamma must have shape (B,H,T)");
    TORCH_CHECK(
        beta.scalar_type() == diagonal_real.scalar_type() &&
            gamma.scalar_type() == diagonal_real.scalar_type(),
        "beta, gamma, and split values must use one dtype");
    c10::cuda::CUDAGuard device_guard(diagonal_real.device());
    const int batch = diagonal_real.size(0);
    const int heads = diagonal_real.size(1);
    const int time = diagonal_real.size(2);
    const int state = diagonal_real.size(3);
    const int dictionary_size = destination.size(1);
    const int rows = batch * heads;
    const int chunks = (time + chunk_size - 1) / chunk_size;
    const size_t shared_bytes = static_cast<size_t>(28) * state;
    auto destination_options = destination.options().dtype(torch::kInt16);
    auto float_options = diagonal_real.options().dtype(torch::kFloat32);
    auto aggregate_destination =
        torch::empty({rows, chunks, state}, destination_options);
    auto aggregate_diagonal_real =
        torch::empty({rows, chunks, state}, float_options);
    auto aggregate_diagonal_imag = torch::empty_like(aggregate_diagonal_real);
    auto aggregate_bias_real = torch::empty_like(aggregate_diagonal_real);
    auto aggregate_bias_imag = torch::empty_like(aggregate_diagonal_real);
    auto output_real = torch::empty_like(diagonal_real);
    auto output_imag = torch::empty_like(diagonal_imag);
    const auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND(
        at::ScalarType::BFloat16,
        diagonal_real.scalar_type(),
        "flash_pd_native_mamba3_local_aggregate",
        [&] {
            if (mode == kPermutationGather) {
                FLASH_PD_LAUNCH(
                    MAMBA3_PHASE_A(scalar_t, kPermutationGather),
                    dim3(rows, chunks),
                    state,
                    shared_bytes,
                    stream,
                    destination.data_ptr<int16_t>(),
                    inverse_destination.data_ptr<int16_t>(),
                    routes.data_ptr<int16_t>(),
                    diagonal_real.data_ptr<scalar_t>(),
                    diagonal_imag.data_ptr<scalar_t>(),
                    value_real.data_ptr<scalar_t>(),
                    value_imag.data_ptr<scalar_t>(),
                    beta.data_ptr<scalar_t>(),
                    gamma.data_ptr<scalar_t>(),
                    aggregate_destination.data_ptr<int16_t>(),
                    aggregate_diagonal_real.data_ptr<float>(),
                    aggregate_diagonal_imag.data_ptr<float>(),
                    aggregate_bias_real.data_ptr<float>(),
                    aggregate_bias_imag.data_ptr<float>(),
                    heads,
                    dictionary_size,
                    time,
                    state,
                    chunks,
                    chunk_size);
            } else {
                FLASH_PD_LAUNCH(
                    MAMBA3_PHASE_A(scalar_t, kGeneralScatter),
                    dim3(rows, chunks),
                    state,
                    shared_bytes,
                    stream,
                    destination.data_ptr<int16_t>(),
                    inverse_destination.data_ptr<int16_t>(),
                    routes.data_ptr<int16_t>(),
                    diagonal_real.data_ptr<scalar_t>(),
                    diagonal_imag.data_ptr<scalar_t>(),
                    value_real.data_ptr<scalar_t>(),
                    value_imag.data_ptr<scalar_t>(),
                    beta.data_ptr<scalar_t>(),
                    gamma.data_ptr<scalar_t>(),
                    aggregate_destination.data_ptr<int16_t>(),
                    aggregate_diagonal_real.data_ptr<float>(),
                    aggregate_diagonal_imag.data_ptr<float>(),
                    aggregate_bias_real.data_ptr<float>(),
                    aggregate_bias_imag.data_ptr<float>(),
                    heads,
                    dictionary_size,
                    time,
                    state,
                    chunks,
                    chunk_size);
            }
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (mode == kPermutationGather) {
        FLASH_PD_LAUNCH(
            MAMBA3_PHASE_B(kPermutationGather),
            rows,
            state,
            shared_bytes,
            stream,
            aggregate_destination.data_ptr<int16_t>(),
            aggregate_diagonal_real.data_ptr<float>(),
            aggregate_diagonal_imag.data_ptr<float>(),
            aggregate_bias_real.data_ptr<float>(),
            aggregate_bias_imag.data_ptr<float>(),
            state,
            chunks);
    } else {
        FLASH_PD_LAUNCH(
            MAMBA3_PHASE_B(kGeneralScatter),
            rows,
            state,
            shared_bytes,
            stream,
            aggregate_destination.data_ptr<int16_t>(),
            aggregate_diagonal_real.data_ptr<float>(),
            aggregate_diagonal_imag.data_ptr<float>(),
            aggregate_bias_real.data_ptr<float>(),
            aggregate_bias_imag.data_ptr<float>(),
            state,
            chunks);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    AT_DISPATCH_FLOATING_TYPES_AND(
        at::ScalarType::BFloat16,
        diagonal_real.scalar_type(),
        "flash_pd_native_mamba3_corrected_replay",
        [&] {
            if (mode == kPermutationGather) {
                FLASH_PD_LAUNCH(
                    MAMBA3_PHASE_C(scalar_t, kPermutationGather),
                    dim3(rows, chunks),
                    state,
                    shared_bytes,
                    stream,
                    destination.data_ptr<int16_t>(),
                    inverse_destination.data_ptr<int16_t>(),
                    routes.data_ptr<int16_t>(),
                    diagonal_real.data_ptr<scalar_t>(),
                    diagonal_imag.data_ptr<scalar_t>(),
                    value_real.data_ptr<scalar_t>(),
                    value_imag.data_ptr<scalar_t>(),
                    beta.data_ptr<scalar_t>(),
                    gamma.data_ptr<scalar_t>(),
                    aggregate_bias_real.data_ptr<float>(),
                    aggregate_bias_imag.data_ptr<float>(),
                    output_real.data_ptr<scalar_t>(),
                    output_imag.data_ptr<scalar_t>(),
                    heads,
                    dictionary_size,
                    time,
                    state,
                    chunks,
                    chunk_size);
            } else {
                FLASH_PD_LAUNCH(
                    MAMBA3_PHASE_C(scalar_t, kGeneralScatter),
                    dim3(rows, chunks),
                    state,
                    shared_bytes,
                    stream,
                    destination.data_ptr<int16_t>(),
                    inverse_destination.data_ptr<int16_t>(),
                    routes.data_ptr<int16_t>(),
                    diagonal_real.data_ptr<scalar_t>(),
                    diagonal_imag.data_ptr<scalar_t>(),
                    value_real.data_ptr<scalar_t>(),
                    value_imag.data_ptr<scalar_t>(),
                    beta.data_ptr<scalar_t>(),
                    gamma.data_ptr<scalar_t>(),
                    aggregate_bias_real.data_ptr<float>(),
                    aggregate_bias_imag.data_ptr<float>(),
                    output_real.data_ptr<scalar_t>(),
                    output_imag.data_ptr<scalar_t>(),
                    heads,
                    dictionary_size,
                    time,
                    state,
                    chunks,
                    chunk_size);
            }
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output_real, output_imag};
}

std::vector<torch::Tensor> flash_pd_native_backward_cuda(
    torch::Tensor destination,
    torch::Tensor routes,
    torch::Tensor diagonal_real,
    torch::Tensor diagonal_imag,
    torch::Tensor output_real,
    torch::Tensor output_imag,
    torch::Tensor grad_output_real,
    torch::Tensor grad_output_imag) {
    c10::cuda::CUDAGuard device_guard(diagonal_real.device());
    const int batch = diagonal_real.size(0);
    const int heads = diagonal_real.size(1);
    const int time = diagonal_real.size(2);
    const int state = diagonal_real.size(3);
    const int dictionary_size = destination.size(1);
    const int rows = batch * heads;
    const size_t shared_bytes = static_cast<size_t>(8) * state;
    auto grad_diagonal_real = torch::empty_like(diagonal_real);
    auto grad_diagonal_imag = torch::empty_like(diagonal_imag);
    auto grad_bias_real = torch::empty_like(diagonal_real);
    auto grad_bias_imag = torch::empty_like(diagonal_imag);
    const auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND(
        at::ScalarType::BFloat16,
        diagonal_real.scalar_type(),
        "flash_pd_native_backward",
        [&] {
            backward_kernel<scalar_t><<<rows, state, shared_bytes, stream>>>(
                destination.data_ptr<int16_t>(),
                routes.data_ptr<int16_t>(),
                diagonal_real.data_ptr<scalar_t>(),
                diagonal_imag.data_ptr<scalar_t>(),
                output_real.data_ptr<scalar_t>(),
                output_imag.data_ptr<scalar_t>(),
                grad_output_real.data_ptr<scalar_t>(),
                grad_output_imag.data_ptr<scalar_t>(),
                grad_diagonal_real.data_ptr<scalar_t>(),
                grad_diagonal_imag.data_ptr<scalar_t>(),
                grad_bias_real.data_ptr<scalar_t>(),
                grad_bias_imag.data_ptr<scalar_t>(),
                heads,
                dictionary_size,
                time,
                state);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        grad_diagonal_real,
        grad_diagonal_imag,
        grad_bias_real,
        grad_bias_imag};
}

std::vector<torch::Tensor> flash_pd_native_paper_backward_cuda(
    torch::Tensor dictionary_logits,
    torch::Tensor selector_logits,
    torch::Tensor destination,
    torch::Tensor routes,
    torch::Tensor diagonal_real,
    torch::Tensor diagonal_imag,
    torch::Tensor bias_real,
    torch::Tensor bias_imag,
    torch::Tensor output_real,
    torch::Tensor output_imag,
    torch::Tensor grad_output_real,
    torch::Tensor grad_output_imag,
    torch::Tensor value_real,
    torch::Tensor value_imag,
    torch::Tensor beta,
    torch::Tensor gamma,
    double dictionary_temperature,
    double router_temperature,
    int64_t chunk_size) {
    TORCH_CHECK(
        dictionary_logits.scalar_type() == torch::kFloat32 &&
            selector_logits.scalar_type() == torch::kFloat32,
        "paper surrogate logits must be float32");
    TORCH_CHECK(
        dictionary_logits.is_cuda() && selector_logits.is_cuda(),
        "paper surrogate logits must be CUDA tensors");
    TORCH_CHECK(
        dictionary_temperature > 0.0 && router_temperature > 0.0,
        "dictionary and router temperatures must be positive");
    TORCH_CHECK(
        chunk_size > 0 && chunk_size <= 128,
        "paper backward chunk_size must be in [1, 128]");
    const int batch = diagonal_real.size(0);
    const int heads = diagonal_real.size(1);
    const int time = diagonal_real.size(2);
    const int state = diagonal_real.size(3);
    const int dictionary_size = destination.size(1);
    const int rows = batch * heads;
    const int chunks = (time + chunk_size - 1) / chunk_size;
    const bool mamba3_siso = value_real.numel() != 0;
    TORCH_CHECK(
        !mamba3_siso ||
            (value_real.sizes() == diagonal_real.sizes() &&
             value_imag.sizes() == diagonal_real.sizes() &&
             beta.sizes() == torch::IntArrayRef({batch, heads, time}) &&
             gamma.sizes() == beta.sizes()),
        "Mamba-3 values must be (B,H,T,N) and beta/gamma must be (B,H,T)");
    TORCH_CHECK(dictionary_size <= 1024, "CUDA paper surrogate requires K <= 1024");
    TORCH_CHECK(
        dictionary_logits.sizes() ==
            torch::IntArrayRef({heads, dictionary_size, state, state}),
        "dictionary logits must have shape (H,K,N,N)");
    TORCH_CHECK(
        selector_logits.sizes() ==
            torch::IntArrayRef({batch, time, heads, dictionary_size}),
        "selector logits must have shape (B,T,H,K)");
    c10::cuda::CUDAGuard device_guard(diagonal_real.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (mamba3_siso) {
        // MAMBA3_FUSED_BACKWARD_BEGIN
        auto grad_diagonal_real = torch::empty_like(diagonal_real);
        auto grad_diagonal_imag = torch::empty_like(diagonal_imag);
        auto grad_value_real = torch::empty_like(value_real);
        auto grad_value_imag = torch::empty_like(value_imag);
        auto grad_beta = torch::empty_like(beta);
        auto grad_gamma = torch::empty_like(gamma);
        auto active_dictionary_gradient = torch::zeros(
            {heads, dictionary_size, state},
            dictionary_logits.options());
        auto grad_dictionary_logits = torch::empty_like(dictionary_logits);
        auto grad_selector_logits = torch::empty_like(selector_logits);
        const bool aggregate_dictionary =
            dictionary_size * state <= 12000 - 2 * state - 32 - 5;
        const int local_dictionary_elements =
            aggregate_dictionary ? dictionary_size * state : 0;
        const size_t fused_shared_bytes =
            static_cast<size_t>(
                2 * state + local_dictionary_elements + 32 + 5) *
            sizeof(float);
        const float router_inverse_temperature =
            1.0f / static_cast<float>(router_temperature);
        AT_DISPATCH_FLOATING_TYPES_AND(
            at::ScalarType::BFloat16,
            diagonal_real.scalar_type(),
            "flash_pd_native_mamba3_backward_fused",
            [&] {
                FLASH_PD_LAUNCH(
                    mamba3_backward_fused_kernel<scalar_t>,
                    rows,
                    state,
                    fused_shared_bytes,
                    stream,
                    selector_logits.data_ptr<float>(),
                    destination.data_ptr<int16_t>(),
                    routes.data_ptr<int16_t>(),
                    diagonal_real.data_ptr<scalar_t>(),
                    diagonal_imag.data_ptr<scalar_t>(),
                    output_real.data_ptr<scalar_t>(),
                    output_imag.data_ptr<scalar_t>(),
                    grad_output_real.data_ptr<scalar_t>(),
                    grad_output_imag.data_ptr<scalar_t>(),
                    value_real.data_ptr<scalar_t>(),
                    value_imag.data_ptr<scalar_t>(),
                    beta.data_ptr<scalar_t>(),
                    gamma.data_ptr<scalar_t>(),
                    grad_diagonal_real.data_ptr<scalar_t>(),
                    grad_diagonal_imag.data_ptr<scalar_t>(),
                    grad_value_real.data_ptr<scalar_t>(),
                    grad_value_imag.data_ptr<scalar_t>(),
                    grad_beta.data_ptr<scalar_t>(),
                    grad_gamma.data_ptr<scalar_t>(),
                    active_dictionary_gradient.data_ptr<float>(),
                    grad_selector_logits.data_ptr<float>(),
                    heads,
                    dictionary_size,
                    time,
                    state,
                    aggregate_dictionary,
                    router_inverse_temperature);
            });
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        const float dictionary_inverse_temperature =
            1.0f / static_cast<float>(dictionary_temperature);
        FLASH_PD_LAUNCH(
            paper_dictionary_gradient_kernel,
            dim3(heads, dictionary_size, state),
            state,
            5 * sizeof(float),
            stream,
            dictionary_logits.data_ptr<float>(),
            destination.data_ptr<int16_t>(),
            active_dictionary_gradient.data_ptr<float>(),
            grad_dictionary_logits.data_ptr<float>(),
            dictionary_size,
            state,
            dictionary_inverse_temperature);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        std::vector<torch::Tensor> gradients = {
            grad_dictionary_logits,
            grad_selector_logits,
            grad_diagonal_real,
            grad_diagonal_imag,
            grad_value_real,
            grad_value_imag,
            grad_beta,
            grad_gamma};
        // MAMBA3_FUSED_BACKWARD_END
        return gradients;
    }

    auto grad_diagonal_real = torch::empty_like(diagonal_real);
    auto grad_diagonal_imag = torch::empty_like(diagonal_imag);
    // These are affine-bias gradients for the baseline and grad_value_real /
    // grad_value_imag for the integrated Mamba-3 SISO recurrence.
    auto grad_value_real = torch::empty_like(bias_real);
    auto grad_value_imag = torch::empty_like(bias_imag);
    auto grad_beta =
        mamba3_siso ? torch::empty_like(beta) : torch::empty({0}, beta.options());
    auto grad_gamma =
        mamba3_siso ? torch::empty_like(gamma) : torch::empty({0}, gamma.options());
    auto active_dictionary_gradient = torch::zeros(
        {heads, dictionary_size, state},
        dictionary_logits.options());
    auto selector_score = torch::empty(
        {rows, time},
        dictionary_logits.options());
    auto destination_options = destination.options().dtype(torch::kInt16);
    auto float_options = dictionary_logits.options().dtype(torch::kFloat32);
    auto aggregate_destination =
        torch::empty({rows, chunks, state}, destination_options);
    auto aggregate_diagonal_real =
        torch::empty({rows, chunks, state}, float_options);
    auto aggregate_diagonal_imag = torch::empty_like(aggregate_diagonal_real);
    auto aggregate_bias_real = torch::empty_like(aggregate_diagonal_real);
    auto aggregate_bias_imag = torch::empty_like(aggregate_diagonal_real);
    auto chunk_carry_real = torch::empty_like(aggregate_diagonal_real);
    auto chunk_carry_imag = torch::empty_like(aggregate_diagonal_real);
    auto grad_dictionary_logits = torch::empty_like(dictionary_logits);
    auto grad_selector_logits = torch::empty_like(selector_logits);
    AT_DISPATCH_FLOATING_TYPES_AND(
        at::ScalarType::BFloat16,
        diagonal_real.scalar_type(),
        "flash_pd_native_paper_backward_local_aggregate",
        [&] {
            paper_backward_phase_a_kernel<scalar_t><<<
                dim3(rows, chunks),
                state,
                static_cast<size_t>(28) * state,
                stream>>>(
                destination.data_ptr<int16_t>(),
                routes.data_ptr<int16_t>(),
                diagonal_real.data_ptr<scalar_t>(),
                diagonal_imag.data_ptr<scalar_t>(),
                grad_output_real.data_ptr<scalar_t>(),
                grad_output_imag.data_ptr<scalar_t>(),
                aggregate_destination.data_ptr<int16_t>(),
                aggregate_diagonal_real.data_ptr<float>(),
                aggregate_diagonal_imag.data_ptr<float>(),
                aggregate_bias_real.data_ptr<float>(),
                aggregate_bias_imag.data_ptr<float>(),
                heads,
                dictionary_size,
                time,
                state,
                chunks,
                chunk_size);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    paper_backward_phase_b_kernel<<<
        rows,
        state,
        static_cast<size_t>(8) * state,
        stream>>>(
        aggregate_destination.data_ptr<int16_t>(),
        aggregate_diagonal_real.data_ptr<float>(),
        aggregate_diagonal_imag.data_ptr<float>(),
        aggregate_bias_real.data_ptr<float>(),
        aggregate_bias_imag.data_ptr<float>(),
        chunk_carry_real.data_ptr<float>(),
        chunk_carry_imag.data_ptr<float>(),
        state,
        chunks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const bool aggregate_dictionary =
        dictionary_size * state <= 12000 - 2 * state - 32;
    const int local_dictionary_elements =
        aggregate_dictionary ? dictionary_size * state : 0;
    const size_t replay_shared_bytes =
        static_cast<size_t>(2 * state + local_dictionary_elements + 32) *
        sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND(
        at::ScalarType::BFloat16,
        diagonal_real.scalar_type(),
        "flash_pd_native_paper_backward_corrected_replay",
        [&] {
            paper_backward_phase_c_kernel<scalar_t><<<
                dim3(rows, chunks),
                state,
                replay_shared_bytes,
                stream>>>(
                destination.data_ptr<int16_t>(),
                routes.data_ptr<int16_t>(),
                diagonal_real.data_ptr<scalar_t>(),
                diagonal_imag.data_ptr<scalar_t>(),
                bias_real.data_ptr<scalar_t>(),
                bias_imag.data_ptr<scalar_t>(),
                output_real.data_ptr<scalar_t>(),
                output_imag.data_ptr<scalar_t>(),
                grad_output_real.data_ptr<scalar_t>(),
                grad_output_imag.data_ptr<scalar_t>(),
                value_real.data_ptr<scalar_t>(),
                value_imag.data_ptr<scalar_t>(),
                beta.data_ptr<scalar_t>(),
                gamma.data_ptr<scalar_t>(),
                chunk_carry_real.data_ptr<float>(),
                chunk_carry_imag.data_ptr<float>(),
                grad_diagonal_real.data_ptr<scalar_t>(),
                grad_diagonal_imag.data_ptr<scalar_t>(),
                grad_value_real.data_ptr<scalar_t>(),
                grad_value_imag.data_ptr<scalar_t>(),
                grad_beta.data_ptr<scalar_t>(),
                grad_gamma.data_ptr<scalar_t>(),
                active_dictionary_gradient.data_ptr<float>(),
                selector_score.data_ptr<float>(),
                heads,
                dictionary_size,
                time,
                state,
                chunks,
                chunk_size,
                aggregate_dictionary,
                mamba3_siso);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const float dictionary_inverse_temperature =
        1.0f / static_cast<float>(dictionary_temperature);
    paper_dictionary_gradient_kernel<<<
        dim3(heads, dictionary_size, state),
        state,
        5 * sizeof(float),
        stream>>>(
        dictionary_logits.data_ptr<float>(),
        destination.data_ptr<int16_t>(),
        active_dictionary_gradient.data_ptr<float>(),
        grad_dictionary_logits.data_ptr<float>(),
        dictionary_size,
        state,
        dictionary_inverse_temperature);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const float router_inverse_temperature =
        1.0f / static_cast<float>(router_temperature);
    paper_selector_gradient_kernel<<<
        dim3(rows, time),
        dictionary_size,
        5 * sizeof(float),
        stream>>>(
        selector_logits.data_ptr<float>(),
        routes.data_ptr<int16_t>(),
        selector_score.data_ptr<float>(),
        grad_selector_logits.data_ptr<float>(),
        heads,
        dictionary_size,
        time,
        router_inverse_temperature);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    std::vector<torch::Tensor> gradients = {
        grad_dictionary_logits,
        grad_selector_logits,
        grad_diagonal_real,
        grad_diagonal_imag,
        grad_value_real,
        grad_value_imag};
    if (mamba3_siso) {
        gradients.push_back(grad_beta);
        gradients.push_back(grad_gamma);
    }
    return gradients;
}
