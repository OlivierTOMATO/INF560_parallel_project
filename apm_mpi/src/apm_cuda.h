#ifndef __APM_CUDA_H_
#define __APM_CUDA_H_

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>

void findMatch(int* local_n_matches, char *buf, int nb_patterns, char **pattern, int start, int end, int n_bytes, int approx_factor);
char *read_input_file(char *filename, int *size);

#endif