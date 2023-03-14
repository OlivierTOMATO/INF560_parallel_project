# INF560-parallel-algorithm

#### Description

In this project, we are targetted at implementing some paralleing algorithms (i.e., OPENMP, MPI, CUDA...) to achieve higher performance in a pattern matching problem in course INF560 Algorithmique parallèle et distribuée. 

We merge all our algorithms together with a decision tree. Finally can achieve a performance increase of around **100**.

The contributors of this project are **Changjie Wang** and **Chinonso Stanislaus Ngwu**

This project is guided by professor **Patrick CARRIBAULT**

#### File architecture

* apm/dna contains the dna database we use for matching

* apm/src contains all the algorithm we implemented. Inside we have

  * mpi directory with all the pure mpi implementation

    * mpi/apm.c is the sequential version
    * mpi/cj_apm_mpi_static.c is the static decomposition
    * mpi/cj_mpi_d.c is the dynamic distribution over file, with pattern together
    * mpi/scan_mpi.c is the dynamic distribution over file, with pattern one by one
    * mpi/scan_mpi_dd.c is the dynamic pattern distribution
    * Makefile to compile the c file into object file to execute

  * openmp directory with all the OpenMP implementation of mpi mentioned before

    * openmp/apm.c is the sequential version
    * openmp/cj_apm_mpi_static.c is the static decomposition with OpenMP over chunks
    * openmp/cj_apm_mpi_static.c is the static decomposition with OpenMP over patterns
    * openmp/cj_apm_mpi_d.c is the dynamic distribution over file with pattern together with OpenMP over chunks
    * openmp/scan.c is the dynamic distribution over file, with pattern one by one, with OpenMP over chunks

    * mpi/scan_mpi_dd.c is the dynamic pattern distributions with OpenMP over chunks
    * Makefile to compile the c file into object file to execute

  * cuda directory gives implementation of cuda

    * cuda/apm.c is the sequential version
    * cuda/cuda_omp_static.cu is the cuda file giving some necessary functions for to run on the cuda devide.
    * cuda/static_decomposition.c is the cuda implementation over static decomposition, with 90% balance between cuda and OpenMP

  * app directory gives the final version of our app merging all the algorithms together, where we provide 12 different decisions finally

    * app/app.c is the final version of our app
    * app/app2.c is the one for test in simulation situation where GPU is ignored
    * app/cuda_omp_static.cu gives the necessary functions to call for a GPU device
    * app/kernel.h gives the header file for cuda_omp_static.cu
    * app/Makefile gives commands to compile
    * app/test.batch gives test batch for decision 1- 4
    * app/test2.batch gives test batch for decision 5-6
    * app/test3.batch gives test batch for decision 7-8
    * app/test4.batch gives test batch for decision 9-12

#### How to run

* Enter app directory

  ```
  apm/src/app
  ```

* Compile

  ```
  make
  ```

* Run batch file

  ```
  sbatch test.batch
  sbatch test2.batch
  sbatch test3.batch
  sbatch test4.batch
  ```

   