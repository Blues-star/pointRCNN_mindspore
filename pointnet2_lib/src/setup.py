from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pointnet2',
    ext_modules=[
        CUDAExtension('pointnet2_cuda', [
            'pointnet2_cuda/pointnet2_api.cpp',
            'pointnet2_cuda/ms_ext.cpp', 
            'pointnet2_cuda/ball_query.cpp', 
            'pointnet2_cuda/ball_query_gpu.cu',
            'pointnet2_cuda/group_points.cpp', 
            'pointnet2_cuda/group_points_gpu.cu',
            'pointnet2_cuda/interpolate.cpp', 
            'pointnet2_cuda/interpolate_gpu.cu',
            'pointnet2_cuda/sampling.cpp', 
            'pointnet2_cuda/sampling_gpu.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)
