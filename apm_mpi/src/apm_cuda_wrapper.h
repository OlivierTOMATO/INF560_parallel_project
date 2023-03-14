#ifndef APM_CUDA_WRAPPER_H
#define APM_CUDA_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

void findMatch(int* local_n_matches, char *buf, int nb_patterns, char **pattern, int start, int end, int n_bytes, int approx_factor);

#ifdef __cplusplus
}
#endif

#endif