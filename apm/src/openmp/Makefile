all:apm cj_apm_mpi_static cj_apm_mpi_static_p cj_apm_mpi_d scan_omp scan_omp_dd

apm: apm.c
	gcc -o apm apm.c

cj_apm_mpi_static:cj_apm_mpi_static.c
	mpicc -fopenmp -o cj_apm_mpi_static cj_apm_mpi_static.c

cj_apm_mpi_static_p:cj_apm_mpi_static_p.c
	mpicc -fopenmp -o cj_apm_mpi_static_p cj_apm_mpi_static_p.c

cj_apm_mpi_d:cj_apm_mpi_d.c
	mpicc -fopenmp  -o cj_apm_mpi_d cj_apm_mpi_d.c

scan_omp:scan_omp.c
	mpicc -fopenmp -o scan_omp scan_omp.c

scan_omp_dd:scan_omp_dd.c
	mpicc -fopenmp -o scan_omp_dd scan_omp_dd.c

clean:
	rm -f apm cj_apm_mpi_static cj_apm_mpi_static_p cj_apm_mpi_d scan_omp scan_omp_dd
	$(foreach file, $(wildcard slurm*), $(RM) $(file))