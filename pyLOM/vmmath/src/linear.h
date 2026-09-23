/*
    Linear operator
*/
#include <complex.h>
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
// Double precision
int dlinear_operator(double *U, double *S, double *VT, double *Atilde, double *Y, double *Z, const double r, const int my, const int mz, const int nn);
void dflip_columns(double *A, double *B, int m, int n);
// Single complex precision
void cflip_columns(scomplex_t *A, scomplex_t *B, int m, int n);
// Double complex precision
void zflip_columns(dcomplex_t *A, dcomplex_t *B, int m, int n);