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
#include <stdbool.h>
#include <omp.h>
#include <mpi.h>

#include "kernel.h"

#define APM_DEBUG 0
#define chunk_size 10000

char *
read_input_file(char *filename, int *size)
{
    char *buf;
    off_t fsize;
    int fd = 0;
    int n_bytes = 1;

    /* Open the text file */
    fd = open(filename, O_RDONLY);
    if (fd == -1)
    {
        fprintf(stderr, "Unable to open the text file <%s>\n", filename);
        return NULL;
    }

    /* Get the number of characters in the textfile */
    fsize = lseek(fd, 0, SEEK_END);
    if (fsize == -1)
    {
        fprintf(stderr, "Unable to lseek to the end\n");
        return NULL;
    }

#if APM_DEBUG
    printf("File length: %lld\n", fsize);
#endif

    /* Go back to the beginning of the input file */
    if (lseek(fd, 0, SEEK_SET) == -1)
    {
        fprintf(stderr, "Unable to lseek to start\n");
        return NULL;
    }

    /* Allocate data to copy the target text */
    buf = (char *)malloc(fsize * sizeof(char));
    if (buf == NULL)
    {
        fprintf(stderr, "Unable to allocate %lld byte(s) for main array\n",
                fsize);
        return NULL;
    }

    n_bytes = read(fd, buf, fsize);
    if (n_bytes != fsize)
    {
        fprintf(stderr,
                "Unable to copy %lld byte(s) from text file (%d byte(s) copied)\n",
                fsize, n_bytes);
        return NULL;
    }

#if APM_DEBUG
    printf("Number of read bytes: %d\n", n_bytes);
#endif

    *size = n_bytes;

    close(fd);

    return buf;
}

#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))

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

// Dyanmic pattern
void decision_1(int rank, int N, int nb_patterns, char *filename, int approx_factor, int n_bytes, int *n_matches, char **pattern, char *buf)
{
    int i, j;
    MPI_Status status;

    /* Timer start */
    double start_time = MPI_Wtime();

    /*****
     * BEGIN MAIN LOOP
     ******/

    if (rank == 0)
    {
        MPI_Request *req = malloc(sizeof(MPI_Request) * nb_patterns);
        int dest_rank;

        for (i = 0; i < nb_patterns; i++)
        {
            MPI_Recv(&dest_rank, 1, MPI_INTEGER, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&i, 1, MPI_INTEGER, dest_rank, 0, MPI_COMM_WORLD);
            MPI_Irecv(&n_matches[i], 1, MPI_INTEGER, dest_rank, 0, MPI_COMM_WORLD, &req[i]);
        }

        /* wait until all the results are received */
        MPI_Waitall(nb_patterns, req, MPI_STATUSES_IGNORE);

        /* send a message that tell the workers to stop */
        for (dest_rank = 1; dest_rank < N; dest_rank++)
        {
            int ready;
            MPI_Recv(&ready, 1, MPI_INTEGER, dest_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int stop_value = -1;
            MPI_Send(&stop_value, 1, MPI_INTEGER, dest_rank, 0, MPI_COMM_WORLD);
        }

        for (i = 0; i < nb_patterns; i++)
        {
            printf("Number of matches for pattern <%s>: %d\n",
                   pattern[i], n_matches[i]);
        }
        /* Timer stop */
        double end_time = MPI_Wtime();
        printf("APM done in %lf s\n\n", end_time - start_time);
    }
    else
    {
        int index;
        while (1)
        {
            MPI_Send(&rank, 1, MPI_INTEGER, 0, 1, MPI_COMM_WORLD);
            MPI_Recv(&index, 1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &status);
            if (index == -1)
            {
                break;
            }

            int size_pattern = strlen(pattern[index]);
            int *column;

            /* Initialize the number of matches to 0 */
            int matches = 0;

            column = (int *)malloc((size_pattern + 1) * sizeof(int));

            /* Traverse the input data up to the end of the file */
            for (j = 0; j < n_bytes; j++)
            {
                int distance = 0;
                int size;

#if APM_DEBUG
                if (j % 100 == 0)
                {
                    printf("Procesing byte %d (out of %d)\n", j, n_bytes);
                }
#endif

                size = size_pattern;
                if (n_bytes - j < size_pattern)
                {
                    size = n_bytes - j;
                }

                distance = levenshtein(pattern[index], &buf[j], size, column);

                if (distance <= approx_factor)
                {
                    matches++;
                }
            }
            MPI_Send(&matches, 1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD);

            free(column);
        }
    }

    free(n_matches);

    /*****
     * END MAIN LOOP
     ******/
}
// dynamic pattern + omp
void decision_2(int rank, int N, int nb_patterns, char *filename, int approx_factor, int n_bytes, int *n_matches, char **pattern, char *buf)
{
    int i, j;
    MPI_Status status;

    /* Timer start */
    double start_time = MPI_Wtime();

    /*****
     * BEGIN MAIN LOOP
     ******/

    if (rank == 0)
    {
        MPI_Request *req = malloc(sizeof(MPI_Request) * nb_patterns);
        int dest_rank;

        for (i = 0; i < nb_patterns; i++)
        {
            MPI_Recv(&dest_rank, 1, MPI_INTEGER, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&i, 1, MPI_INTEGER, dest_rank, 0, MPI_COMM_WORLD);
            MPI_Irecv(&n_matches[i], 1, MPI_INTEGER, dest_rank, 0, MPI_COMM_WORLD, &req[i]);
        }

        /* wait until all the results are received */
        MPI_Waitall(nb_patterns, req, MPI_STATUSES_IGNORE);

        /* send a message that tell the workers to stop */
        for (dest_rank = 1; dest_rank < N; dest_rank++)
        {
            int ready;
            MPI_Recv(&ready, 1, MPI_INTEGER, dest_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int stop_value = -1;
            MPI_Send(&stop_value, 1, MPI_INTEGER, dest_rank, 0, MPI_COMM_WORLD);
        }

        for (i = 0; i < nb_patterns; i++)
        {
            printf("Number of matches for pattern <%s>: %d\n",
                   pattern[i], n_matches[i]);
        }
        /* Timer stop */
        double end_time = MPI_Wtime();
        printf("APM done in %lf s\n\n", end_time - start_time);
    }
    else
    {
        int index;
        while (1)
        {
            MPI_Send(&rank, 1, MPI_INTEGER, 0, 1, MPI_COMM_WORLD);
            MPI_Recv(&index, 1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &status);
            if (index == -1)
            {
                break;
            }

            int size_pattern = strlen(pattern[index]);
            int *column;
            int size = size_pattern;
            int distance;

            /* Initialize the number of matches to 0 */
            int local_num = 0;
#pragma omp parallel private(j, distance, column) firstprivate(size)
            {
                column = (int *)malloc((size_pattern + 1) * sizeof(int));
                // if (column == NULL)
                // {
                //     fprintf(stderr, "Error: unable to allocate memory for column (%ldB)\n",
                //             (size_pattern + 1) * sizeof(int));
                //     return 1;
                // }

#pragma omp for reduction(+ \
                          : local_num) schedule(dynamic)
                /* Traverse the input data up to the end of the file */
                for (j = 0; j < n_bytes; j++)
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

                    distance = levenshtein(pattern[index], &buf[j], size, column);

                    if (distance <= approx_factor)
                    {
                        local_num++;
                    }
                }
            }
            MPI_Send(&local_num, 1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD);
        }
    }
}
// dynamic pattern + gpu
void decision_3(int rank, int N, int nb_patterns, char *filename, int approx_factor, int n_bytes, int *n_matches, char **pattern, char *buf)
{
    int i, j;
    MPI_Status status;

    /* Timer start */
    double start_time = MPI_Wtime();

    /*****
     * BEGIN MAIN LOOP
     ******/

    if (rank == 0)
    {
        MPI_Request *req = malloc(sizeof(MPI_Request) * nb_patterns);

        int dest_rank;

        for (i = 0; i < nb_patterns; i++)
        {
            MPI_Recv(&dest_rank, 1, MPI_INTEGER, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&i, 1, MPI_INTEGER, dest_rank, 0, MPI_COMM_WORLD);
            MPI_Irecv(&n_matches[i], 1, MPI_INTEGER, dest_rank, 0, MPI_COMM_WORLD, &req[i]);
        }

        /* wait until all the results are received */
        MPI_Waitall(nb_patterns, req, MPI_STATUSES_IGNORE);

        /* send a message that tell the workers to stop */
        for (dest_rank = 1; dest_rank < N; dest_rank++)
        {
            int ready;
            MPI_Recv(&ready, 1, MPI_INTEGER, dest_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int stop_value = -1;
            MPI_Send(&stop_value, 1, MPI_INTEGER, dest_rank, 0, MPI_COMM_WORLD);
        }

        for (i = 0; i < nb_patterns; i++)
        {
            printf("Number of matches for pattern <%s>: %d\n",
                   pattern[i], n_matches[i]);
        }
        /* Timer stop */
        double end_time = MPI_Wtime();
        printf("APM done in %lf s\n\n", end_time - start_time);
    }
    else
    {
        int index;
        int nbGPU;
        cudaGetDeviceCount(&nbGPU);
        cudaSetDevice(rank % nbGPU);
        while (1)
        {
            MPI_Send(&rank, 1, MPI_INTEGER, 0, 1, MPI_COMM_WORLD);
            MPI_Recv(&index, 1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &status);
            if (index == -1)
            {
                break;
            }

            int *local_match;
            local_match = (int *)malloc(sizeof(int));

            findMatch(local_match, buf, 1, &pattern[index], 0, n_bytes, n_bytes, approx_factor, 1);

            /*****
             * END MAIN LOOP
             ******/
            MPI_Send(local_match, 1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD);
        }
    }
}
// dynamic pattern + omp + gpu
void decision_4(int rank, int N, int nb_patterns, char *filename, int approx_factor, int n_bytes, int *n_matches, char **pattern, char *buf)
{
    int i, j;
    MPI_Status status;

    /* Timer start */
    double start_time = MPI_Wtime();

    /*****
     * BEGIN MAIN LOOP
     ******/

    if (rank == 0)
    {
        MPI_Request *req = malloc(sizeof(MPI_Request) * nb_patterns);
        int dest_rank;

        for (i = 0; i < nb_patterns; i++)
        {
            MPI_Recv(&dest_rank, 1, MPI_INTEGER, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&i, 1, MPI_INTEGER, dest_rank, 0, MPI_COMM_WORLD);
            MPI_Irecv(&n_matches[i], 1, MPI_INTEGER, dest_rank, 0, MPI_COMM_WORLD, &req[i]);
        }

        /* wait until all the results are received */
        MPI_Waitall(nb_patterns, req, MPI_STATUSES_IGNORE);

        /* send a message that tell the workers to stop */
        for (dest_rank = 1; dest_rank < N; dest_rank++)
        {
            int ready;
            MPI_Recv(&ready, 1, MPI_INTEGER, dest_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int stop_value = -1;
            MPI_Send(&stop_value, 1, MPI_INTEGER, dest_rank, 0, MPI_COMM_WORLD);
        }

        for (i = 0; i < nb_patterns; i++)
        {
            printf("Number of matches for pattern <%s>: %d\n",
                   pattern[i], n_matches[i]);
        }
        /* Timer stop */
        double end_time = MPI_Wtime();
        printf("APM done in %lf s\n\n", end_time - start_time);
    }
    else
    {
        int index;
        int nbGPU;
        cudaGetDeviceCount(&nbGPU);
        cudaSetDevice(rank % nbGPU);
        while (1)
        {
            MPI_Send(&rank, 1, MPI_INTEGER, 0, 1, MPI_COMM_WORLD);
            MPI_Recv(&index, 1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &status);
            if (index == -1)
            {
                break;
            }

            int *local_match;
            local_match = (int *)malloc(sizeof(int));

            findMatch(local_match, buf, 1, &pattern[index], 0, n_bytes, n_bytes, approx_factor, 0.95);

            /*****
             * END MAIN LOOP
             ******/
            MPI_Send(local_match, 1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD);
        }
    }
}
// static
void decision_9(int rank, int N, int nb_patterns, char *filename, int approx_factor, int n_bytes, int *n_matches, char **pattern, char *buf)
{
    int i, j;
    /* Timer start */
    double start_time = MPI_Wtime();
    int start_point = rank * (n_bytes / N + (n_bytes % N > 0));
    int end_point = (rank + 1) * (n_bytes / N + (n_bytes % N > 0));

    if (end_point > n_bytes)
    {
        end_point = n_bytes;
    }

    n_matches = (int *)malloc(nb_patterns * sizeof(int));
    int *local_n_matches = (int *)malloc((nb_patterns) * sizeof(int));

    /*****
     * BEGIN MAIN LOOP
     ******/
    /* Check each pattern one by one */
    for (i = 0; i < nb_patterns; i++)
    {
        int size_pattern = strlen(pattern[i]);
        int *column;

        /* Initialize the number of matches to 0 */
        local_n_matches[i] = 0;

        column = (int *)malloc((size_pattern + 1) * sizeof(int));
        if (column == NULL)
        {
            fprintf(stderr, "Error: unable to allocate memory for column (%ldB)\n",
                    (size_pattern + 1) * sizeof(int));
        }

        /* Traverse the input data up to the end of the file */
        for (j = start_point; j < end_point; j++)
        {
            int distance = 0;
            int size;

#if APM_DEBUG
            if (j % 100 == 0)
            {
                printf("Procesing byte %d (out of %d)\n", j, n_bytes);
            }
#endif

            size = size_pattern;
            if (n_bytes - j < size_pattern)
            {
                size = n_bytes - j;
            }

            distance = levenshtein(pattern[i], &buf[j], size, column);

            if (distance <= approx_factor)
            {
                local_n_matches[i]++;
            }
        }
        free(column);
    }

    /*****
     * END MAIN LOOP
     ******/
    MPI_Reduce(local_n_matches, n_matches, nb_patterns, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        /* Timer stop */
        double end_time = MPI_Wtime();

        for (i = 0; i < nb_patterns; i++)
        {
            printf("Number of matches for pattern <%s>: %d\n",
                   pattern[i], n_matches[i]);
        }

        printf("APM done in %lf s\n\n", end_time - start_time);
    }
}
// dynamic scan
void decision_7(int rank, int N, int nb_patterns, char *filename, int approx_factor, int n_bytes, int *n_matches, char **pattern, char *buf)
{
    int p, i, j;
    MPI_Status status;

    /* Timer start */
    double start_time = MPI_Wtime();
    /*****
     * BEGIN MAIN LOOP
     ******/

    /* Check each pattern one by one */
    for (p = 0; p < nb_patterns; p++)
    {
        int local_match = 0;
        if (rank == 0)
        {
            // MPI_Request *req = malloc(freq * sizeof(MPI_Request));
            int dest_rank;

            for (i = 0; i < n_bytes; i += chunk_size)
            {
                MPI_Recv(&dest_rank, 1, MPI_INTEGER, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                MPI_Send(&i, 1, MPI_INTEGER, dest_rank, 0, MPI_COMM_WORLD);
            }

            // MPI_Waitall(freq, req, MPI_STATUSES_IGNORE);

            /* send a message that tell the workers to stop */
            for (dest_rank = 1; dest_rank < N; dest_rank++)
            {
                int ready;
                MPI_Recv(&ready, 1, MPI_INTEGER, dest_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                int stop_value = -1;
                MPI_Send(&stop_value, 1, MPI_INTEGER, dest_rank, 0, MPI_COMM_WORLD);
            }
        }
        else
        {
            // printf("In rank %d\n", rank);
            int index;
            int start;
            int end;
            while (1)
            {
                MPI_Send(&rank, 1, MPI_INTEGER, 0, 1, MPI_COMM_WORLD);
                MPI_Recv(&index, 1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &status);
                if (index == -1)
                {
                    break;
                }
                int size_pattern = strlen(pattern[p]);
                int *column;

                /* Initialize the number of matches to 0 */

                column = (int *)malloc((size_pattern + 1) * sizeof(int));
                if (column == NULL)
                {
                    fprintf(stderr, "Error: unable to allocate memory for column (%ldB)\n",
                            (size_pattern + 1) * sizeof(int));
                }

                end = index + chunk_size;
                if (end >= n_bytes)
                {
                    end = n_bytes;
                }
                start = index;

                /* Traverse the input data up to the end of the file */
                for (j = start; j < end; j++)
                {
                    int distance = 0;
                    int size;

#if APM_DEBUG
                    if (j % 100 == 0)
                    {
                        printf("Procesing byte %d (out of %d)\n", j, n_bytes);
                    }
#endif

                    size = size_pattern;
                    if (n_bytes - j < size_pattern)
                    {
                        size = n_bytes - j;
                    }

                    distance = levenshtein(pattern[p], &buf[j], size, column);

                    if (distance <= approx_factor)
                    {
                        local_match++;
                    }
                }

                free(column);
            }
        }
        MPI_Reduce(&local_match, &n_matches[p], 1, MPI_INTEGER, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if (rank == 0)
    {

        for (i = 0; i < nb_patterns; i++)
        {
            printf("Number of matches for pattern <%s>: %d\n",
                   pattern[i], n_matches[i]);
        }
        /* Timer stop */
        double end_time = MPI_Wtime();
        printf("APM done in %lf s\n\n", end_time - start_time);
    }
}
// dynamic cj
void decision_5(int rank, int N, int nb_patterns, char *filename, int approx_factor, int n_bytes, int *n_matches, char **pattern, char *buf)
{
    int i, j;
    /* Timer start */
    double start_time = MPI_Wtime();
    int *local_n_matches = (int *)malloc(nb_patterns * sizeof(int));
    for (i = 0; i < nb_patterns; i++)
    {
        n_matches[i] = 0;
        local_n_matches[i] = 0;
    }

    MPI_Status status;
    // the variable to set
    int freq = n_bytes / chunk_size + (n_bytes % chunk_size > 0);
    // char hostname[1024];
    // gethostname(hostname, 1024);

    // printf("processor_name: %s\n", hostname);

    if (rank == 0)
    {
        int dest_rank;

        for (i = 0; i < n_bytes; i += chunk_size)
        {
            MPI_Recv(&dest_rank, 1, MPI_INTEGER, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Send(&i, 1, MPI_INTEGER, dest_rank, 0, MPI_COMM_WORLD);
        }

        /* send a message that tell the workers to stop */
        for (dest_rank = 1; dest_rank < N; dest_rank++)
        {
            int ready;
            MPI_Recv(&ready, 1, MPI_INTEGER, dest_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int stop_value = -1;
            MPI_Send(&stop_value, 1, MPI_INTEGER, dest_rank, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        int index;
        int start;
        int end;
        while (1)
        {

            MPI_Send(&rank, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Recv(&index, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            if (index == -1)
            {
                break;
            }

            /* Check each pattern one by one */
            for (i = 0; i < nb_patterns; i++)
            {
                int size_pattern = strlen(pattern[i]);
                int *column;

                /* Initialize the number of matches to 0 */

                column = (int *)malloc((size_pattern + 1) * sizeof(int));
                if (column == NULL)
                {
                    fprintf(stderr, "Error: unable to allocate memory for column (%ldB)\n",
                            (size_pattern + 1) * sizeof(int));
                }

                end = index + chunk_size;
                if (end >= n_bytes)
                {
                    end = n_bytes;
                }
                start = index;

                // if (index == 0)
                // {
                //     start = index;
                // }
                // else
                // {
                //     start = index - size_pattern + 1;
                // }

                /* Traverse the input data up to the end of the file */
                for (j = start; j < end; j++)
                {
                    int distance = 0;
                    int size;

#if APM_DEBUG
                    if (j % 100 == 0)
                    {
                        printf("Procesing byte %d (out of %d)\n", j, n_bytes);
                    }
#endif

                    size = size_pattern;
                    if (n_bytes - j < size_pattern)
                    {
                        size = n_bytes - j;
                    }

                    distance = levenshtein(pattern[i], &buf[j], size, column);

                    if (distance <= approx_factor)
                    {
                        local_n_matches[i]++;
                    }
                }

                free(column);
            }
        }
    }

    MPI_Reduce(local_n_matches, n_matches, nb_patterns, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        /* Timer stop */
        double end_time = MPI_Wtime();

        for (i = 0; i < nb_patterns; i++)
        {
            printf("Number of matches for pattern <%s>: %d\n",
                   pattern[i], n_matches[i]);
        }
        printf("APM done in %lf s\n\n", end_time - start_time);
    }

    free(local_n_matches);
    free(n_matches);
}
// static + omp
void decision_10(int rank, int N, int nb_patterns, char *filename, int approx_factor, int n_bytes, int *n_matches, char **pattern, char *buf)
{
    /* Timer start */
    double start_time = MPI_Wtime();
    int i, j;
    int start_point = rank * (n_bytes / N + (n_bytes % N > 0));
    int end_point = (rank + 1) * (n_bytes / N + (n_bytes % N > 0));

    if (end_point > n_bytes)
    {
        end_point = n_bytes;
    }

    n_matches = (int *)malloc(nb_patterns * sizeof(int));
    int *local_n_matches = (int *)malloc((nb_patterns) * sizeof(int));
    /*****
     * BEGIN MAIN LOOP
     ******/

    /* Check each pattern one by one */
    /* Implementtation of opnemp */
    // #pragma omp for schedule(static)
    for (i = 0; i < nb_patterns; i++)
    {
        int size_pattern = strlen(pattern[i]);
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
                          : local_num) schedule(dynamic)
            for (j = start_point; j < end_point; j++)
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

        local_n_matches[i] = local_num;
    }

    /*****
     * END MAIN LOOP
     ******/
    MPI_Reduce(local_n_matches, n_matches, nb_patterns, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        /* Timer stop */
        double end_time = MPI_Wtime();

        for (i = 0; i < nb_patterns; i++)
        {
            printf("Number of matches for pattern <%s>: %d\n",
                   pattern[i], n_matches[i]);
        }

        printf("APM done in %lf s\n\n", end_time - start_time);
    }
}
// dynamic scan + omp
void decision_8(int rank, int N, int nb_patterns, char *filename, int approx_factor, int n_bytes, int *n_matches, char **pattern, char *buf)
{
    int i, j, p;
    MPI_Status status;

    /* Timer start */
    double start_time = MPI_Wtime();

    /*****
     * BEGIN MAIN LOOP
     ******/

    /* Check each pattern one by one */
    for (p = 0; p < nb_patterns; p++)
    {
        int local_match = 0;
        if (rank == 0)
        {
            int dest_rank;

            for (i = 0; i < n_bytes; i += chunk_size)
            {
                MPI_Recv(&dest_rank, 1, MPI_INTEGER, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                MPI_Send(&i, 1, MPI_INTEGER, dest_rank, 0, MPI_COMM_WORLD);
            }

            /* send a message that tell the workers to stop */
            for (dest_rank = 1; dest_rank < N; dest_rank++)
            {
                int ready;
                MPI_Recv(&ready, 1, MPI_INTEGER, dest_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                int stop_value = -1;
                MPI_Send(&stop_value, 1, MPI_INTEGER, dest_rank, 0, MPI_COMM_WORLD);
            }
        }
        else
        {
            // printf("In rank %d\n", rank);
            int index;
            int start;
            int end;

            while (1)
            {
                MPI_Send(&rank, 1, MPI_INTEGER, 0, 1, MPI_COMM_WORLD);
                MPI_Recv(&index, 1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &status);
                if (index == -1)
                {
                    break;
                }
                int size_pattern = strlen(pattern[p]);
                int *column;
                int distance;
                int size = size_pattern;
                int local_num = 0;

                /* Initialize the number of matches to 0 */
                /* Traverse the input data up to the end of the file */

                end = index + chunk_size;
                if (end >= n_bytes)
                {
                    end = n_bytes;
                }
                start = index;

#pragma omp parallel private(j, distance, column) firstprivate(size)
                {
                    column = (int *)malloc((size_pattern + 1) * sizeof(int));
#pragma omp for schedule(static) reduction(+ \
                                           : local_match)
                    /* Traverse the input data up to the end of the file */
                    for (j = start; j < end; j++)
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

                        distance = levenshtein(pattern[p], &buf[j], size, column);

                        if (distance <= approx_factor)
                        {
                            local_match++;
                        }
                    }

                    free(column);
                }
            }
        }
        MPI_Reduce(&local_match, &n_matches[p], 1, MPI_INTEGER, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if (rank == 0)
    {
        for (i = 0; i < nb_patterns; i++)
        {
            printf("Number of matches for pattern <%s>: %d\n",
                   pattern[i], n_matches[i]);
        }
        /* Timer stop */
        double end_time = MPI_Wtime();
        printf("APM done in %lf s\n\n", end_time - start_time);
    }
}
// dynamic cj + omp
void decision_6(int rank, int N, int nb_patterns, char *filename, int approx_factor, int n_bytes, int *n_matches, char **pattern, char *buf)
{
    /* Timer start */
    double start_time = MPI_Wtime();
    int i, j;
    n_matches = (int *)malloc(nb_patterns * sizeof(int));
    int *local_n_matches = (int *)malloc((nb_patterns) * sizeof(int));
    MPI_Status status;
    // the variable to set
    int freq = n_bytes / chunk_size + (n_bytes % chunk_size > 0);

    if (rank == 0)
    {
        int dest_rank;

        for (i = 0; i < n_bytes; i += chunk_size)
        {
            MPI_Recv(&dest_rank, 1, MPI_INTEGER, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Send(&i, 1, MPI_INTEGER, dest_rank, 0, MPI_COMM_WORLD);
        }

        /* send a message that tell the workers to stop */
        for (dest_rank = 1; dest_rank < N; dest_rank++)
        {
            int ready;
            MPI_Recv(&ready, 1, MPI_INTEGER, dest_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int stop_value = -1;
            MPI_Send(&stop_value, 1, MPI_INTEGER, dest_rank, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        int index;
        int start;
        int end;
        while (1)
        {

            MPI_Send(&rank, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Recv(&index, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            if (index == -1)
            {
                break;
            }

            /* Check each pattern one by one */
            for (i = 0; i < nb_patterns; i++)
            {
                int size_pattern = strlen(pattern[i]);
                int size = size_pattern;
                /* Initialize the number of matches to 0 */
                int local_num = 0;
                int *column;
                int distance;

                end = index + chunk_size;
                if (end >= n_bytes)
                {
                    end = n_bytes;
                }
                start = index;

/* Traverse the input data up to the end of the file */
#pragma omp parallel private(j, distance, column) firstprivate(size)
                {
                    column = (int *)malloc((size_pattern + 1) * sizeof(int));
#pragma omp for reduction(+ \
                          : local_num)
                    for (j = start; j < end; j++)
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
                local_n_matches[i] += local_num;
            }
        }
    }

    MPI_Reduce(local_n_matches, n_matches, nb_patterns, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        /* Timer stop */
        double end_time = MPI_Wtime();

        for (i = 0; i < nb_patterns; i++)
        {
            printf("Number of matches for pattern <%s>: %d\n",
                   pattern[i], n_matches[i]);
        }
        printf("APM done in %lf s\n\n", end_time - start_time);
    }

    free(local_n_matches);
    free(n_matches);
}
// static + gpu
void decision_11(int rank, int N, int nb_patterns, char *filename, int approx_factor, int n_bytes, int *n_matches, char **pattern, char *buf)
{
    int i;
    /* Timer start */
    double start_time = MPI_Wtime();
    int start_point = rank * (n_bytes / N + (n_bytes % N > 0));
    int end_point = (rank + 1) * (n_bytes / N + (n_bytes % N > 0));

    if (end_point > n_bytes)
    {
        end_point = n_bytes;
    }

    // printf("start index: %d, end index: %d\n", start_point, end_point);

    n_matches = (int *)malloc(nb_patterns * sizeof(int));
    int *local_n_matches = (int *)malloc((nb_patterns) * sizeof(int));
    /*****
     * BEGIN MAIN LOOP
     ******/
    int nbGPU;
    cudaGetDeviceCount(&nbGPU);
    cudaSetDevice(rank % nbGPU);
    findMatch(local_n_matches, buf, nb_patterns, pattern, start_point, end_point, n_bytes, approx_factor, 1);

    /*****
     * END MAIN LOOP
     ******/
    MPI_Reduce(local_n_matches, n_matches, nb_patterns, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        /* Timer stop */
        double end_time = MPI_Wtime();

        for (i = 0; i < nb_patterns; i++)
        {
            printf("Number of matches for pattern <%s>: %d\n",
                   pattern[i], n_matches[i]);
        }
        printf("APM done in %lf s\n\n", end_time - start_time);
    }
}
// static + omp + gpu
void decision_12(int rank, int N, int nb_patterns, char *filename, int approx_factor, int n_bytes, int *n_matches, char **pattern, char *buf)
{
    int i;
    /* Timer start */
    double start_time = MPI_Wtime();
    int start_point = rank * (n_bytes / N + (n_bytes % N > 0));
    int end_point = (rank + 1) * (n_bytes / N + (n_bytes % N > 0));

    if (end_point > n_bytes)
    {
        end_point = n_bytes;
    }

    // printf("start index: %d, end index: %d\n", start_point, end_point);

    n_matches = (int *)malloc(nb_patterns * sizeof(int));
    int *local_n_matches = (int *)malloc((nb_patterns) * sizeof(int));
    /*****
     * BEGIN MAIN LOOP
     ******/
    int nbGPU;
    cudaGetDeviceCount(&nbGPU);
    cudaSetDevice(rank % nbGPU);
    findMatch(local_n_matches, buf, nb_patterns, pattern, start_point, end_point, n_bytes, approx_factor, 0.95);

    /*****
     * END MAIN LOOP
     ******/
    MPI_Reduce(local_n_matches, n_matches, nb_patterns, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        /* Timer stop */
        double end_time = MPI_Wtime();

        for (i = 0; i < nb_patterns; i++)
        {
            printf("Number of matches for pattern <%s>: %d\n",
                   pattern[i], n_matches[i]);
        }
        printf("APM done in %lf s\n\n", end_time - start_time);
    }
}

int main(int argc, char **argv)
{
    char **pattern;
    char *filename;
    int approx_factor = 0;
    int nb_patterns = 0;
    int i, j;
    char *buf;
    struct timeval t1, t2;
    double duration;
    int n_bytes;
    int *n_matches;

    /* Check number of arguments */
    if (argc < 4)
    {
        printf("Usage: %s approximation_factor "
               "dna_database pattern1 pattern2 ...\n",
               argv[0]);
        return 1;
    }

    /* Get the distance factor */
    approx_factor = atoi(argv[1]);

    /* Grab the filename containing the target text */
    filename = argv[2];

    /* Get the number of patterns that the user wants to search for */
    nb_patterns = argc - 3;

    /* Fill the pattern array */
    pattern = (char **)malloc(nb_patterns * sizeof(char *));
    if (pattern == NULL)
    {
        fprintf(stderr,
                "Unable to allocate array of pattern of size %d\n",
                nb_patterns);
        return 1;
    }

    /* Grab the patterns */
    for (i = 0; i < nb_patterns; i++)
    {
        int l;

        l = strlen(argv[i + 3]);
        if (l <= 0)
        {
            fprintf(stderr, "Error while parsing argument %d\n", i + 3);
            return 1;
        }

        pattern[i] = (char *)malloc((l + 1) * sizeof(char));
        if (pattern[i] == NULL)
        {
            fprintf(stderr, "Unable to allocate string of size %d\n", l);
            return 1;
        }

        strncpy(pattern[i], argv[i + 3], (l + 1));
    }

    buf = read_input_file(filename, &n_bytes);
    if (buf == NULL)
    {
        return 1;
    }

    /* Allocate the array of matches */
    n_matches = (int *)malloc(nb_patterns * sizeof(int));
    if (n_matches == NULL)
    {
        fprintf(stderr, "Error: unable to allocate memory for %ldB\n",
                nb_patterns * sizeof(int));
        return 1;
    }

    /*****
     * BEGIN MAIN LOOP
     ******/

    /* Timer start */
    gettimeofday(&t1, NULL);

    int rank, N;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &N);

    if (rank == 0)
    {
        printf("parallel Pattern Matching code\n");
        printf("Approximate Pattern Mathing: "
               "looking for %d pattern(s) in file %s w/ distance of %d\n",
               nb_patterns, filename, approx_factor);
    }
    int decision = 0;
    if (rank == 0)
    {

        bool OMP = true;
        bool GPU = true;
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        GPU = deviceCount > 0;
        int num_threads = 0;

#pragma omp parallel
        {
            num_threads = omp_get_num_threads();
            // printf("Number of threads in OpenMP parallel region: %d\n", num_threads);
        }

        OMP = num_threads > 1;
        if (OMP)
        {
            omp_set_num_threads(2);
        }
        printf("GPU: %d OMP: %d\n", GPU, num_threads);

        printf("%s\n", "We are assuming that one will never input patterns more than 20, otherwise it makes our app look redundant");
        if (nb_patterns % N == 0)
        {
            printf("%s\n", "OK, we plan to dynamically allocate your pattern to different process we have");
            decision = 0;
            if (!GPU)
            {
                if (OMP)
                {
                    printf("%s\n", "given that you don't have a GPU and have multiple openmp threads");
                    printf("%s\n", "our final decision is to implement MPI + OMP over Dynamic Pattern Distribution");
                    decision = 2;
                }
                else
                {
                    printf("%s\n", "given that you don't have a GPU and have single openmp thread");
                    printf("%s\n", "our final decision is to implement pure MPI over Dynamic Pattern Distribution");
                    decision = 1;
                }
            }
            else
            {
                if (OMP)
                {
                    printf("%s\n", "given that you have a GPU and have multiple openmp threads");
                    printf("%s\n", "our final decision is to implement MPI with each process running hybrid OMP + CUDA, over Dynamic Pattern Distribution");
                    decision = 4;
                }
                else
                {
                    printf("%s\n", "given that you have a GPU and have single openmp thread");
                    printf("%s\n", "our final decision is to implement MPI with each process running CUDA, over Dynamic Pattern Distribution");
                    decision = 3;
                }
            }
        }
        else
        {
            if (GPU)
            {
                if (OMP)
                {
                    printf("%s\n", "given that you have a GPU and have multiple openmp threads");
                    printf("%s\n", "our final decision is to implement MPI with each process running hybrid OMP + CUDA, over Static Decomposition");
                    decision = 12;
                }
                else
                {
                    printf("%s\n", "given that you have a GPU and have multiple openmp threads");
                    printf("%s\n", "our final decision is to implement MPI with each process running CUDA, over Static Decomposition");
                    decision = 11;
                }
            }
            else
            {
                if (N < 16)
                {
                    printf("%s\n", "OK, you don't have a GPU, but it seems that you don't have enough ranks, let's keep using static mpi");
                    if (OMP)
                    {
                        printf("%s\n", "given that you don't have a GPU and have multiple openmp threads");
                        printf("%s\n", "our final decision is to implement MPI with each process running OMP, over Static Decomposition");
                        decision = 10;
                    }
                    else
                    {
                        printf("%s\n", "given that you don't have a GPU and have single openmp thread");
                        printf("%s\n", "our final decision is to implement pure MPI, over Static Decomposition");
                        decision = 9;
                    }
                }
                else
                {
                    printf("%s\n", "OK. Even though you don't have a GPU, it seems that you have enough ranks, we plan to divide your test database into small chunks and allocate them to the processes");
                    if (n_bytes > 12582912)
                    {
                        printf("%s\n", "Hello, are you still there? This is a big file, we suggest that we deal with the pattern one by one");
                        if (OMP)
                        {
                            printf("%s\n", "given that you don't have a GPU and have multiple openmp threads");
                            printf("%s\n", "our final decision is to implement MPI with each process running OMP, over Dynamic Distribution with Patterns One by One");
                            decision = 8;
                        }
                        else
                        {
                            printf("%s\n", "given that you don't have a GPU and have single openmp thread");
                            printf("%s\n", "our final decision is to implement pure MPI, over Dynamic Distribution with Patterns One by One");
                            decision = 7;
                        }
                    }
                    else
                    {
                        printf("%s\n", "Hello, it seems that the database you gave us is not that large, ask the process find the matches for all the pattern together");
                        if (OMP)
                        {
                            printf("%s\n", "given that you don't have a GPU and have multiple openmp threads");
                            printf("%s\n", "our final decision is to implement MPI with each process running OMP, over Dynamic Distribution Patterns together");
                            decision = 6;
                        }
                        else
                        {
                            printf("%s\n", "given that you don't have a GPU and have single openmp thread");
                            printf("%s\n", "our final decision is to implement pure MPI, over Dynamic Distribution Patterns together");
                            decision = 5;
                        }
                    }
                }
            }
        }
        printf("Our final decision is to use: decision %d\n", decision);
        for (i = 0; i < N; i++)
        {
            MPI_Send(&decision, 1, MPI_INTEGER, i, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(&decision, 1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    switch (decision)
    {
    case 1:
        decision_1(rank, N, nb_patterns, filename, approx_factor, n_bytes, n_matches, pattern, buf);
        break;
    case 2:
        decision_2(rank, N, nb_patterns, filename, approx_factor, n_bytes, n_matches, pattern, buf);
        break;
    case 3:
        decision_3(rank, N, nb_patterns, filename, approx_factor, n_bytes, n_matches, pattern, buf);
        break;
    case 4:
        decision_4(rank, N, nb_patterns, filename, approx_factor, n_bytes, n_matches, pattern, buf);
        break;
    case 5:
        decision_5(rank, N, nb_patterns, filename, approx_factor, n_bytes, n_matches, pattern, buf);
        break;
    case 6:
        decision_6(rank, N, nb_patterns, filename, approx_factor, n_bytes, n_matches, pattern, buf);
        break;
    case 7:
        decision_7(rank, N, nb_patterns, filename, approx_factor, n_bytes, n_matches, pattern, buf);
        break;
    case 8:
        decision_8(rank, N, nb_patterns, filename, approx_factor, n_bytes, n_matches, pattern, buf);
        break;
    case 9:
        decision_9(rank, N, nb_patterns, filename, approx_factor, n_bytes, n_matches, pattern, buf);
        break;
    case 10:
        decision_10(rank, N, nb_patterns, filename, approx_factor, n_bytes, n_matches, pattern, buf);
        break;
    case 11:
        decision_11(rank, N, nb_patterns, filename, approx_factor, n_bytes, n_matches, pattern, buf);
        break;
    case 12:
        decision_12(rank, N, nb_patterns, filename, approx_factor, n_bytes, n_matches, pattern, buf);
        break;
    }

   
    MPI_Finalize();

    return 0;
}