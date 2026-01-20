/*
	SVD - Singular Value Decomposition of a matrix
*/
#include <complex.h>
typedef float  _Complex scomplex_t;
typedef double _Complex dcomplex_t;
#ifdef USE_MKL
#define MKL_Complex8  scomplex_t
#define MKL_Complex16 dcomplex_t
#include "mkl.h"
#endif
// Single precision version
int ssvd(float *U, float *S, float *VT, float *Y, const int m, const int n);
int stsqr_svd(float *Ui, float *S, float *VT, float *Ai, const int m, const int n);
int srandomized_svd(float *Ui, float *S, float *VT, float *Ai, const int m, const int n, const int r, const int q, unsigned int seed);
// Double precision version
int dsvd(double *U, double *S, double *VT, double *Y, const int m, const int n);
int dtsqr_svd(double *Ui, double *S, double *VT, double *Ai, const int m, const int n);
int drandomized_svd(double *Ui, double *S, double *VT, double *Ai, const int m, const int n, const int r, const int q, unsigned int seed);
// Single complex version
int csvd(scomplex_t *U, float *S, scomplex_t *VT, scomplex_t *Y, const int m, const int n);
int ctsqr_svd(scomplex_t *Ui, float *S, scomplex_t *VT, scomplex_t *Ai, const int m, const int n);
// Double complex version
int zsvd(dcomplex_t *U, double *S, dcomplex_t *VT, dcomplex_t *Y, const int m, const int n);
int ztsqr_svd(dcomplex_t *Ui, double *S, dcomplex_t *VT, dcomplex_t *Ai, const int m, const int n);
