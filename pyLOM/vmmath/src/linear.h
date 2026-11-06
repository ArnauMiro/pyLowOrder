/*
    Linear operator
*/
#include <complex.h>
#include <stdbool.h>
typedef float _Complex scomplex_t;
typedef double _Complex dcomplex_t;
#ifdef USE_MKL
#define MKL_Complex8 scomplex_t
#define MKL_Complex16 dcomplex_t
#include "mkl.h"
#endif
// Single precision
int slinear_operator(float *U, float *S, float *VT, float *Atilde, float *Y, float *Z, const float r, const int my, const int mz, const int nn);
void sflip_columns(float *A, float *B, int m, int n);
void sconcatenate(float *Y, float *Z, float **X, int *n_list, const int n_matrix, const int m, const int n_total, int remove_mean);
void sseparate(float *Y, float *Z, float *X, const int m, const int n, int remove_mean);
// Double precision
int dlinear_operator(double *U, double *S, double *VT, double *Atilde, double *Y, double *Z, const double r, const int my, const int mz, const int nn);
void dflip_columns(double *A, double *B, int m, int n);
void dconcatenate(double *Y, double *Z, double **X, int *n_list, const int n_matrix, const int m, const int n_total, int remove_mean);
void dseparate(double *Y, double *Z, double *X, const int m, const int n, int remove_mean);
// Single complex precision
void cflip_columns(scomplex_t *A, scomplex_t *B, int m, int n);
void cresolvent(scomplex_t *U, float *S, scomplex_t *V, float *A, scomplex_t w, const int n);
// Double complex precision
void zflip_columns(dcomplex_t *A, dcomplex_t *B, int m, int n);
void zresolvent(dcomplex_t *U, double *S, dcomplex_t *V, double *A, dcomplex_t w, const int n);