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
#include <omp.h>

#include "kernel.h"

#define APM_DEBUG 0

int __device__ min3(int a, int b, int c)
{
    return ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)));
}

#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))


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

int levenshtein(char *s1, char *s2, int len, int *column)
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
            column[y] = MIN3(
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
        if (n_bytes - (j + start) < size_pattern)
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

extern "C" void findMatch(int *local_n_matches, char *buf, int nb_patterns, char **pattern, int start, int end, int n_bytes, int approx_factor)
{
    int *d_num;
    char *d_pattern;
    int num;
    int i;
    char *d_buf;
    double percentage = 0.9;
    int end_1 = start + (end - start) * percentage;
    int start_1 = end_1 + 1;

    printf("%d %d %d %d\n", start, end_1, start_1, end);

    cudaMalloc((void **)&d_buf, (end - start + 100) * sizeof(char));
    cudaMemcpy(d_buf, &buf[start], end - start + 100, cudaMemcpyHostToDevice);
    for (i = 0; i < nb_patterns; i++)
    {
        int size_pattern = strlen(pattern[i]);

        cudaMalloc((void **)&d_num, sizeof(int));
        cudaMalloc((void **)&d_pattern, size_pattern * sizeof(char));
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaMemcpyAsync(d_pattern, pattern[i], size_pattern * sizeof(char), cudaMemcpyHostToDevice);
        /* Traverse the input data up to the end of the file */
        matches_kernel<<<4, 256, 0, stream>>>(d_buf, d_pattern, d_num, size_pattern, start, end_1, n_bytes, approx_factor);

        int j;
        int size = size_pattern;
        /* Initialize the number of matches to 0 */
        int local_num = 0;
        local_n_matches[i] = 0;
        int *column;
        int distance;

/* Traverse the input data up to the end of the file */
/* CJ: note here, the column should be set private for each thread, and initialized.
    Use reduction to avoid race condition
    With this, should increase the efficiency more than 50%
*/
#pragma omp parallel private(j, distance, column) firstprivate(size)
        {
            column = (int *)malloc((size_pattern + 1) * sizeof(int));
#pragma omp for reduction(+ \
                          : local_num)
            for (j = start_1; j < end; j++)
            {
                distance = 0;

#if APM_DEBUG
                if (j % 100 == 0)
                {
                    printf("Procesing byte %d (out of %d)\n", j, n_bytes);
                }
#endif
                if (n_bytes - j < size_pattern)
                {
                    size = n_bytes - j;
                }

                distance = levenshtein(pattern[i], &buf[j], size, column);

                if (distance <= approx_factor)
                {
                    local_num++;
                }
            }
            free(column);
        }

        cudaMemcpyAsync(&local_n_matches[i], d_num, sizeof(int), cudaMemcpyDeviceToHost);

        cudaStreamSynchronize(stream);

        local_n_matches[i] += local_num;
        // printf("start: %d, end: %d :matches %d\n", start, end, local_n_matches[i]);
        cudaFree(d_pattern);
        cudaFree(d_num);
    }

}