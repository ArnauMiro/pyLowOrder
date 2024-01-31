/*
	Vector and matrix math operations
*/
#include <complex.h>
typedef double _Complex complex_t;
#ifdef USE_MKL
#define MKL_Complex16 complex_t
#include "mkl.h"
#endif
// Double version
void   transpose(double *A, double *B, const int m, const int n);
double vector_norm(double *v, int start, int n);
void   reorder(double *A, int m, int n, int N);
void   matmult(double *C, double *A, double *B, const int m, const int n, const int k, const char *TA, const char *TB);
void   matmul(double *C, double *A, double *B, const int m, const int n, const int k);
void   matmulp(double *C, double *A, double *B, const int m, const int n, const int k);
void   vecmat(double *v, double *A, const int m, const int n);
int    inverse(double *A, int N, char *UoL);
double RMSE(double *A, double *B, const int m, const int n, MPI_Comm comm);
void   sort(double *v, int *index, int n);
// Double complex version
void   zmatmult(complex_t *C, complex_t *A, complex_t *B, const int m, const int n, const int k, const char *TA, const char *TB);
void   zmatmul(complex_t *C, complex_t *A, complex_t *B, const int m, const int n, const int k);
void   zmatmulp(complex_t *C, complex_t *A, complex_t *B, const int m, const int n, const int k);
void   zvecmat(complex_t *v, complex_t *A, const int m, const int n);
int    zinverse(complex_t *A, int N, char *UoL);
int    eigen(double *real, double *imag, complex_t *vecs, double *A, const int m, const int n);
int    cholesky(complex_t *A, int N);
void   vandermonde(complex_t *Vand, double *real, double *imag, int m, int n);
void   vandermondeTime(complex_t *Vand, double *real, double *imag, int m, int n, double *t);
void   zsort(complex_t *v, int *index, int n);
