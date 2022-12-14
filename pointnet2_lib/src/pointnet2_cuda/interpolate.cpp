#include <torch/serialize/tensor.h>
#include <vector>
//#include <THC/THC.h>
#include <c10/cuda/CUDAStream.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "interpolate_gpu.h"
#include "ms_ext.h"

//extern THCState *state;


void three_nn_wrapper_fast(int b, int n, int m, at::Tensor unknown_tensor, 
    at::Tensor known_tensor, at::Tensor dist2_tensor, at::Tensor idx_tensor) {
    const float *unknown = unknown_tensor.data_ptr<float>();
    const float *known = known_tensor.data_ptr<float>();
    float *dist2 = dist2_tensor.data_ptr<float>();
    int *idx = idx_tensor.data_ptr<int>();

    //cudaStream_t stream = THCState_getCurrentStream(state);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(); 
    three_nn_kernel_launcher_fast(b, n, m, unknown, known, dist2, idx, stream);
}

extern "C" int ms_three_nn_wrapper_fast(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra){
    auto tensors = get_torch_tensors(nparam, params, ndims, shapes, dtypes, c10::kCUDA);
    auto b = tensors[0].item<int>();
    auto n = tensors[1].item<int>();
    auto m = tensors[2].item<int>();

    three_nn_wrapper_fast(b,n,m,tensors[3],tensors[4],tensors[5],tensors[6]);
    return 0;
}


void three_interpolate_wrapper_fast(int b, int c, int m, int n,
                         at::Tensor points_tensor,
                         at::Tensor idx_tensor,
                         at::Tensor weight_tensor,
                         at::Tensor out_tensor) {

    const float *points = points_tensor.data_ptr<float>();
    const float *weight = weight_tensor.data_ptr<float>();
    float *out = out_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();

    //cudaStream_t stream = THCState_getCurrentStream(state);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(); 
    three_interpolate_kernel_launcher_fast(b, c, m, n, points, idx, weight, out, stream);
}

extern "C" int ms_three_interpolate_wrapper_fast(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra){
    auto tensors = get_torch_tensors(nparam, params, ndims, shapes, dtypes, c10::kCUDA);
    auto b = tensors[0].item<int>();
    auto c = tensors[1].item<int>();
    auto m = tensors[2].item<int>();
    auto n = tensors[3].item<int>();

    three_interpolate_wrapper_fast(b,c,m,n,tensors[4],tensors[5],tensors[6],tensors[7]);
    return 0;
}


void three_interpolate_grad_wrapper_fast(int b, int c, int n, int m,
                            at::Tensor grad_out_tensor,
                            at::Tensor idx_tensor,
                            at::Tensor weight_tensor,
                            at::Tensor grad_points_tensor) {

    const float *grad_out = grad_out_tensor.data_ptr<float>();
    const float *weight = weight_tensor.data_ptr<float>();
    float *grad_points = grad_points_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();

    //cudaStream_t stream = THCState_getCurrentStream(state);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(); 
    three_interpolate_grad_kernel_launcher_fast(b, c, n, m, grad_out, idx, weight, grad_points, stream);
}

extern "C" int ms_three_interpolate_grad_wrapper_fast(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra){
    auto tensors = get_torch_tensors(nparam, params, ndims, shapes, dtypes, c10::kCUDA);
    auto b = tensors[0].item<int>();
    auto c = tensors[1].item<int>();
    auto n = tensors[2].item<int>();
    auto m = tensors[3].item<int>();
    three_interpolate_grad_wrapper_fast(b,c,m,n,tensors[4],tensors[5],tensors[6],tensors[7]);
    return 0;
}