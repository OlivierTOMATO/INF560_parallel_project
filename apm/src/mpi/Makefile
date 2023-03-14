all:apm scan_mpi scan_mpi_dd cj_apm_mpi_d cj_apm_mpi_static

apm: apm.c
	gcc -o apm apm.c

scan_mpi:scan_mpi.c
	mpicc  -o scan_mpi scan_mpi.c

scan_mpi_dd:scan_mpi_dd.c
	mpicc  -o scan_mpi_dd scan_mpi_dd.c

cj_apm_mpi_d:cj_apm_mpi_d.c
	mpicc   -o cj_apm_mpi_d cj_apm_mpi_d.c

cj_apm_mpi_static:cj_apm_mpi_static.c
	mpicc  -o cj_apm_mpi_static cj_apm_mpi_static.c

clean:
	rm -f apm scan_mpi scan_mpi_dd cj_apm_mpi cj_apm_mpi_d cj_apm_mpi_static
	$(foreach file, $(wildcard slurm*), $(RM) $(file))