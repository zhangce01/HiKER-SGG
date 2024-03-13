all: roi_align_kernel.cu roi_align_kernel.h
	/usr/local/cuda-9.0/bin/nvcc -c -o roi_align.cu.o roi_align_kernel.cu --compiler-options -fPIC \
    -gencode arch=compute_37,code=sm_37 \
    -gencode arch=compute_52,code=sm_52 \
    -gencode arch=compute_60,code=sm_60 \
    -gencode arch=compute_61,code=sm_61 \
    -gencode arch=compute_70,code=sm_70 
clean:
	rm roi_align.cu.o
