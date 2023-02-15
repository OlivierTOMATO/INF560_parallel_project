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
// input mpi

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

    
    int rank, N;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &N);

    /* Timer start */
    double start_time = MPI_Wtime();
    
    if(rank == 0){
        printf("Approximate Pattern Mathing: "
            "looking for %d pattern(s) in file %s w/ distance of %d\n",
            nb_patterns, filename, approx_factor);
    }

    buf = read_input_file(filename, &n_bytes);
    if (buf == NULL)
    {
        return 1;
    }

    /* Allocate the array of matches */
    n_matches = (int *)malloc(nb_patterns * sizeof(int));
    int *local_n_matches = (int *)malloc(nb_patterns * sizeof(int));

    if (n_matches == NULL)
    {
        fprintf(stderr, "Error: unable to allocate memory for %ldB\n",
                nb_patterns * sizeof(int));
        return 1;
    }

    for (i = 0; i < nb_patterns; i++)
    {
        n_matches[i] = 0;
        local_n_matches[i] = 0;
    }

    /*****
     * BEGIN MAIN LOOP
     ******/

    MPI_Status status;
    // the variable to set
    printf("file size: %d", n_bytes);
    int chunk_size = 100000;
    int freq = n_bytes / chunk_size + (n_bytes % chunk_size > 0);
    MPI_Request *req = malloc(freq * sizeof(MPI_Request));

    if (rank == 0)
    {
        int num = 0;
        for (i = 0; i < n_bytes; i += chunk_size)
        {
            int over;
            int dst;
            MPI_Recv(&dst, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
            MPI_Send(&i, 1, MPI_INT, dst, 0, MPI_COMM_WORLD);
            MPI_Irecv(&over, 1, MPI_INT, dst, 0, MPI_COMM_WORLD, &req[i / chunk_size]);
        }

        MPI_Waitall(freq, req, MPI_STATUSES_IGNORE);

        for (i = 1; i < N; i++)
        {
            int dst;
            MPI_Recv(&dst, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
            int stop = -1;
            MPI_Send(&stop, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
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
                    return 1;
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

                int over = 1;
                MPI_Send(&over, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
        }
    }

    MPI_Reduce(local_n_matches, n_matches, nb_patterns, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        for (i = 0; i < nb_patterns; i++)
        {
            printf("Number of matches for pattern <%s>: %d\n",
                   pattern[i], n_matches[i]);
        }
        /* Timer stop */
        double end_time = MPI_Wtime();
        printf("APM done in %lf s\n", end_time - start_time);
    }

    MPI_Finalize();
    free(local_n_matches);
    free(n_matches);

    /*****
     * END MAIN LOOP
     ******/

    return 0;
}