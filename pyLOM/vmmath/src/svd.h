/*
	SVD - Singular Value Decomposition of a matrix
*/
#include <complex.h>
typedef double _Complex complex_t;
#ifdef USE_MKL
#define MKL_Complex16 complex_t
#include "mkl.h"
#endif
// Double precision version
int qr(double *Q, double *R, double *A, const int m, const int n);
int svd(double *U, double *S, double *VT, double *Y, const int m, const int n);
int tsqr(double *Qi, double *R, double *Ai, const int m, const int n, MPI_Comm comm);
int tsqr_svd(double *Ui, double *S, double *VT, double *Ai, const int m, const int n, MPI_Comm comm);
// Double complex version
int zqr(complex_t *Q, complex_t *R, complex_t *A, const int m, const int n);
int zsvd(complex_t *U, double *S, complex_t *VT, complex_t *Y, const int m, const int n);
int ztsqr(complex_t *Qi, complex_t *R, complex_t *Ai, const int m, const int n, MPI_Comm comm);
int ztsqr_svd(complex_t *Ui, double *S, complex_t *VT, complex_t *Ai, const int m, const int n, MPI_Comm comm);
