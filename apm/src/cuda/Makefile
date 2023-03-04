all:apm cuda_omp_static cuda_static_decomposition

apm: apm.c
	gcc -o apm apm.c

# cuda_static_decomposition:cuda_static_decomposition.c
# 	nvcc -o kernel kernel.cu 
# 	mpicc  -o cuda_static_decomposition  cuda_static_decomposition.c
# 	mpicc cuda_static_decomposition kernel -L/usr/local/cuda/lib64/ -lcudart -o kernel_exe

CUDA_LIB := /usr/local/cuda/lib64/
CUDA_LIBS := -lcudart

cuda_omp_static: cuda_omp_static.cu
	nvcc -Xcompiler -fopenmp -o cuda_omp_static -c cuda_omp_static.cu

cuda_static_decomposition: cuda_static_decomposition.c cuda_omp_static
	mpicc -o cuda_static_decomposition -c cuda_static_decomposition.c
	mpicc -fopenmp cuda_omp_static cuda_static_decomposition -L$(CUDA_LIB) $(CUDA_LIBS) -lcudart -o run



clean:
	rm -f apm cuda_static_decomposition kernel_exe kernel cuda_omp_static
	$(foreach file, $(wildcard slurm*), $(RM) $(file))