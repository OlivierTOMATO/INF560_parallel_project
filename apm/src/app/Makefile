all:cuda_omp_static app

# app:app.c
# 	nvcc -o kernel kernel.cu 
# 	mpicc  -o app  app.c
# 	mpicc app kernel -L/usr/local/cuda/lib64/ -lcudart -o kernel_exe

CUDA_LIB := /usr/local/cuda/lib64/
CUDA_LIBS := -lcudart

cuda_omp_static: cuda_omp_static.cu
	nvcc -Xcompiler -fopenmp -o cuda_omp_static -c cuda_omp_static.cu

app: app.c cuda_omp_static
	mpicc -fopenmp -o app -c app.c
	mpicc -fopenmp cuda_omp_static app -L$(CUDA_LIB) $(CUDA_LIBS) -lcudart -o run



clean:
	rm -f app kernel cuda_omp_static run
	$(foreach file, $(wildcard slurm*), $(RM) $(file))