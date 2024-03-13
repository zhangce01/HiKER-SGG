import os
import torch
from torch.utils.cpp_extension import BuildExtension
from setuptools import setup


sources = []
headers = []
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/nms_cuda.c']
    headers += ['src/nms_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
extra_objects = ['src/cuda/nms.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = BuildExtension(
    '_ext.nms',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects
)

if __name__ == '__main__':
    setup(
        name='_ext.nms',
        headers=headers,
        sources=sources,
        define_macros=defines,
        relative_to=__file__,
        with_cuda=with_cuda,
        extra_objects=extra_objects
        ext_modules=[
            CUDAExtension(
                    name='cuda_extension',
                    sources=['extension.cpp', 'extension_kernel.cu'],
                    extra_compile_args={'cxx': ['-g'],
                                        'nvcc': ['-O2']})
        ],
        cmdclass={
            'build_ext': BuildExtension
        })
