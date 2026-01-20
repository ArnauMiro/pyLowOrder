/*
	QR - QR ecomposition of a matrix
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
int sqr(float *Q, float *R, float *A, const int m, const int n);
int stsqr(float *Qi, float *R, float *Ai, const int m, const int n);
int sqr_streaming(float *Qi, float *B, float *Y, float *Ai, const int m, const int n, const int r, const int q, unsigned int seed);
int slocal_qr_streaming(float *Qi, float *B, float *Y, float *Ai, const int m, const int n, const int r, const int q, unsigned int seed);
int srandomized_qr(float *Qi, float *B, float *Ai, const int m, const int n, const int r, const int q, unsigned int seed);
int slocal_randomized_qr(float *Qi, float *B, float *Ai, const int m, const int n, const int r, const int q, unsigned int seed);
int sinit_randomized_qr(float *Qi, float *B, float *Y, float *Ai, const int m, const int n, const int r, const int q, unsigned int seed);
int supdate_randomized_qr(float *Q2, float *B2, float *Yn, float *Q1, float *B1, float *Yo, float *Ai, const int m, const int n, const int n1, const int n2, const int r, const int q, unsigned int seed);
// Double precision version
int dqr(double *Q, double *R, double *A, const int m, const int n);
int dtsqr(double *Qi, double *R, double *Ai, const int m, const int n);
int drandomized_qr(double *Qi, double *B, double *Ai, const int m, const int n, const int r, const int q, unsigned int seed);
int dlocal_randomized_qr(double *Qi, double *B, double *Ai, const int m, const int n, const int r, const int q, unsigned int seed);
int dinit_randomized_qr(double *Qi, double *B, double *Y, double *Ai, const int m, const int n, const int r, const int q, unsigned int seed);
int dupdate_randomized_qr(double *Q2, double *B2, double *Yn, double *Q1, double *B1, double *Yo, double *Ai, const int m, const int n, const int n1, const int n2, const int r, const int q, unsigned int seed);
// Single complex version
int cqr(scomplex_t *Q, scomplex_t *R, scomplex_t *A, const int m, const int n);
int ctsqr(scomplex_t *Qi, scomplex_t *R, scomplex_t *Ai, const int m, const int n);
// Double complex version
int zqr(dcomplex_t *Q, dcomplex_t *R, dcomplex_t *A, const int m, const int n);
int ztsqr(dcomplex_t *Qi, dcomplex_t *R, dcomplex_t *Ai, const int m, const int n);