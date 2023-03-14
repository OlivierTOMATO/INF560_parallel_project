/**
 * APPROXIMATE PATTERN MATCHING
 *
 * INF560
 */
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>

#include "kernel.h"

#define APM_DEBUG 0

int __device__ min3(int a, int b, int c)
{
    return ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)));
}

int __device__ levenshtein_cuda(char *s1, char *s2, int len, int *column)
{
    unsigned int x, y, lastdiag, olddiag;

    for (y = 1; y <= len; y++)
    {
        column[y] = y;
    }
    for (x = 1; x <= len; x++)
    {
        column[0] = x;
        lastdiag = x - 1;
        for (y = 1; y <= len; y++)
        {
            olddiag = column[y];
            column[y] = min3(
                column[y] + 1,
                column[y - 1] + 1,
                lastdiag + (s1[y - 1] == s2[x - 1] ? 0 : 1));
            lastdiag = olddiag;
        }
    }
    return (column[len]);
}

void __global__ matches_kernel(char *d_buf, char *d_pattern, int *d_num, int size_pattern, int start, int end, int n_bytes, int approx_factor)
{

    /* Traverse the input data up to the end of the file */
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int distance = 0;
    int size;
    int num_local = 0;

    size = size_pattern;
    int *columns = (int *)malloc((size_pattern + 1) * sizeof(int));
    while (j < end - start)
    {
        if (n_bytes - (j + start)< size_pattern)
        {
            size = n_bytes - j - start;
        }

        distance = levenshtein_cuda(d_pattern, &d_buf[j], size, columns);
        if (distance <= approx_factor)
        {
            atomicAdd(&d_num[0], 1);
        }
        j += stride;
    }
    free(columns);
}


extern "C" void findMatch(int* local_n_matches, char *buf, int nb_patterns, char **pattern, int start, int end, int n_bytes, int approx_factor){
    int *d_num;
    char *d_pattern;
    int num;
    int i;
    char *d_buf;

    cudaMalloc((void **)&d_buf, (end - start + 100) * sizeof(char));
    cudaMemcpy(d_buf, &buf[start], end - start + 100, cudaMemcpyHostToDevice);
    for (i = 0; i < nb_patterns; i++)
    { 
        int size_pattern = strlen(pattern[i]);

        cudaMalloc((void **)&d_num, sizeof(int));
        cudaMalloc((void **)&d_pattern, size_pattern * sizeof(char));
        cudaMemcpy(d_pattern, pattern[i],  size_pattern * sizeof(char), cudaMemcpyHostToDevice);

        /* Traverse the input data up to the end of the file */
        matches_kernel<<<4, 256>>>(d_buf, d_pattern, d_num, size_pattern, start, end, n_bytes, approx_factor);

        cudaMemcpy(&local_n_matches[i], d_num, sizeof(int), cudaMemcpyDeviceToHost);
        // printf("start: %d, end: %d :matches %d\n", start, end, local_n_matches[i]);
        cudaFree(d_pattern);
        cudaFree(d_num);
    }
}