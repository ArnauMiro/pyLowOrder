/*
	Vector and matrix math operations
*/
#include <complex.h>
#include "mkl.h"
typedef double _Complex complex_t;
#define MKL_Complex16 complex_t
void   transpose(double *A, double *B, const int m, const int n);
double vector_norm(double *v, int start, int n);
void   reorder(double *A, int m, int n, int N);
void   matmul(double *C, double *A, double *B, const int m, const int n, const int k);
void   matmul_paral(double *C, double *A, double *B, const int m, const int n, const int k);
void   matmul_complex(complex_t *C, complex_t *A, complex_t *B, const int m, const int n, const int k, char *TransA, char *TransB);
void   vecmat(double *v, double *A, const int m, const int n);
void   vecmat_complex(complex_t *v, complex_t *A, const int m, const int n);
int    eigen(double *real, double *imag, complex_t *vecs, double *A, const int m, const int n);
double RMSE(double *A, double *B, const int m, const int n, MPI_Comm comm);
int    cholesky(complex_t *A, int N);
void   vandermonde(complex_t *Vand, double *real, double *imag, int m, int n);
void   vandermondeTime(complex_t *Vand, double *real, double *imag, int m, int n, double *t);
int    inverse(complex_t *A, int N, int UoL);
void   index_sort(double *v, int *index, int n);
