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
int sqr(float *Q, float *R, float *A, const int m, const int n);
int ssvd(float *U, float *S, float *VT, float *Y, const int m, const int n);
int stsqr(float *Qi, float *R, float *Ai, const int m, const int n, MPI_Comm comm);
int stsqr_svd(float *Ui, float *S, float *VT, float *Ai, const int m, const int n, MPI_Comm comm);
int srandomized_qr(float *Qi, float *B, float *Ai, const int m, const int n, const int r, const int q, unsigned int seed, MPI_Comm comm);
int srandomized_svd(float *Ui, float *S, float *VT, float *Ai, const int m, const int n, const int r, const int q, unsigned int seed, MPI_Comm comm);
// Double precision version
int dqr(double *Q, double *R, double *A, const int m, const int n);
int dsvd(double *U, double *S, double *VT, double *Y, const int m, const int n);
int dtsqr(double *Qi, double *R, double *Ai, const int m, const int n, MPI_Comm comm);
int dtsqr_svd(double *Ui, double *S, double *VT, double *Ai, const int m, const int n, MPI_Comm comm);
int drandomized_qr(double *Qi, double *B, double *Ai, const int m, const int n, const int r, const int q, unsigned int seed, MPI_Comm comm);
int drandomized_svd(double *Ui, double *S, double *VT, double *Ai, const int m, const int n, const int r, const int q, unsigned int seed, MPI_Comm comm);
// Single complex version
int cqr(scomplex_t *Q, scomplex_t *R, scomplex_t *A, const int m, const int n);
int csvd(scomplex_t *U, float *S, scomplex_t *VT, scomplex_t *Y, const int m, const int n);
int ctsqr(scomplex_t *Qi, scomplex_t *R, scomplex_t *Ai, const int m, const int n, MPI_Comm comm);
int ctsqr_svd(scomplex_t *Ui, float *S, scomplex_t *VT, scomplex_t *Ai, const int m, const int n, MPI_Comm comm);
// Double complex version
int zqr(dcomplex_t *Q, dcomplex_t *R, dcomplex_t *A, const int m, const int n);
int zsvd(dcomplex_t *U, double *S, dcomplex_t *VT, dcomplex_t *Y, const int m, const int n);
int ztsqr(dcomplex_t *Qi, dcomplex_t *R, dcomplex_t *Ai, const int m, const int n, MPI_Comm comm);
int ztsqr_svd(dcomplex_t *Ui, double *S, dcomplex_t *VT, dcomplex_t *Ai, const int m, const int n, MPI_Comm comm);
