#include <torch/serialize/tensor.h>
#include <vector>
//#include <THC/THC.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "ball_query_gpu.h"
#include "ms_ext.h"

//extern THCState *state;

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int ball_query_wrapper_fast(int b, int n, int m, float radius, int nsample, 
    at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor) {
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(xyz_tensor);
    const float *new_xyz = new_xyz_tensor.data_ptr<float>();
    const float *xyz = xyz_tensor.data_ptr<float>();
    int *idx = idx_tensor.data_ptr<int>();
    cudaMemset(idx, 0, b*m*nsample*sizeof(int));
    //      idx: (B, M, nsample)
    
    // cudaStream_t stream = THCState_getCurrentStream(state);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    ball_query_kernel_launcher_fast(b, n, m, radius, nsample, new_xyz, xyz, idx, stream);
    return 1;
}

extern "C" int ms_ball_query_wrapper_fast(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra){
    auto tensors = get_torch_tensors(nparam, params, ndims, shapes, dtypes, c10::kCUDA);
    auto b = tensors[0].item<int>();
    auto n = tensors[1].item<int>();
    auto m = tensors[2].item<int>();
    auto radius = tensors[3].item<float>();
    auto nsample = tensors[4].item<int>();
    fprintf(stderr, "b,n,m,r,nsample %d %d %d %f %d \n", b, n, m, radius, nsample);
    ball_query_wrapper_fast(b,n,m,radius,nsample,tensors[5],tensors[6],tensors[7]);
    return 0;
}
