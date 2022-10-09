#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>
//#include <THC/THC.h>

#include "sampling_gpu.h"
#include "ms_ext.h"

//extern THCState *state;


int gather_points_wrapper_fast(int b, int c, int n, int npoints, 
    at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor){
    const float *points = points_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();
    float *out = out_tensor.data_ptr<float>();

    //cudaStream_t stream = THCState_getCurrentStream(state);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(); 
    gather_points_kernel_launcher_fast(b, c, n, npoints, points, idx, out, stream);
    return 1;
}

extern "C" int ms_gather_points_wrapper_fast(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra){
    auto tensors = get_torch_tensors(nparam, params, ndims, shapes, dtypes, c10::kCUDA);
    auto b = tensors[0].item<int>();
    auto c = tensors[1].item<int>();
    auto n = tensors[2].item<int>();
    auto npoints = tensors[3].item<int>();
    //fprintf(stderr, "ms_gather_points_wrapper_fast b c n npoints %d %d %d %d \n", b, c, n, npoints);
    gather_points_wrapper_fast(b,c,n,npoints,tensors[4],tensors[5],tensors[6]);
    return 0;
}


int gather_points_grad_wrapper_fast(int b, int c, int n, int npoints, 
    at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor grad_points_tensor) {

    const float *grad_out = grad_out_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();
    float *grad_points = grad_points_tensor.data_ptr<float>();

    //cudaStream_t stream = THCState_getCurrentStream(state);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(); 
    gather_points_grad_kernel_launcher_fast(b, c, n, npoints, grad_out, idx, grad_points, stream);
    return 1;
}

extern "C" int ms_gather_points_grad_wrapper_fast(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra){
    auto tensors = get_torch_tensors(nparam, params, ndims, shapes, dtypes, c10::kCUDA);
    auto b = tensors[0].item<int>();
    auto c = tensors[1].item<int>();
    auto n = tensors[2].item<int>();
    auto npoints = tensors[3].item<int>();

    gather_points_grad_wrapper_fast(b,c,n,npoints,tensors[4],tensors[5],tensors[6]);
    return 0;
}


int furthest_point_sampling_wrapper(int b, int n, int m, 
    at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor) {

    const float *points = points_tensor.data_ptr<float>();
    float *temp = temp_tensor.data_ptr<float>();
    int *idx = idx_tensor.data_ptr<int>();

    //cudaStream_t stream = THCState_getCurrentStream(state);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(); 
    furthest_point_sampling_kernel_launcher(b, n, m, points, temp, idx, stream);
    return 1;
}


extern "C" int ms_furthest_point_sampling_wrapper(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra){
    auto tensors = get_torch_tensors(nparam, params, ndims, shapes, dtypes, c10::kCUDA);
    auto b = tensors[0].item<int>();
    auto n = tensors[1].item<int>();
    auto m = tensors[2].item<int>();

    furthest_point_sampling_wrapper(b,n,m,tensors[3],tensors[4],tensors[5]);
    return 0;
}