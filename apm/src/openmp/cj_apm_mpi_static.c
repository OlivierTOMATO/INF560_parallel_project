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
#include <mpi.h>
#include <omp.h>

#define APM_DEBUG 0

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

    /* Timer start */
    double start_time = MPI_Wtime();

    int rank, N;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &N);

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
        return 1.0;
    }

    /* Grab the patterns */
    for (i = 0; i < nb_patterns; i++)
    {
        int l;

        l = strlen(argv[i + 3]);
        if (l <= 0)
        {
            fprintf(stderr, "Error while parsing argument %d\n", i + 3);
            return 1.0;
        }

        pattern[i] = (char *)malloc((l + 1) * sizeof(char));
        if (pattern[i] == NULL)
        {
            fprintf(stderr, "Unable to allocate string of size %d\n", l);
            return 1.0;
        }

        strncpy(pattern[i], argv[i + 3], (l + 1));
    }

    if (rank == 0)
    {
        printf("MPI + OPENMP: \nMPI: Divide the record file into average size (size == file size / the number of ranks) and Allocate it directly to all the ranks (including the rank 0)\n OPENMP: The chunk received by each rank is run in parallel in all the available threads\n");

        printf("Approximate Pattern Mathing: "
               "looking for %d pattern(s) in file %s w/ distance of %d (function called: mpi_data_split)\n",
               nb_patterns, filename, approx_factor);

    }

    /* Every process loads the data */
    buf = read_input_file(filename, &n_bytes);
    if (buf == NULL)
    {
        MPI_Abort(MPI_COMM_WORLD, 0);
        return 1.0;
    }

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

    MPI_Finalize();
    return 0;
}