/*
	Vector and matrix math operations
*/
#include <complex.h>
typedef float  _Complex scomplex_t;
typedef double _Complex dcomplex_t;
#ifdef USE_MKL
#define MKL_Complex8  scomplex_t
#define MKL_Complex16 dcomplex_t
#include "mkl.h"
#endif
// Float version
void   stranspose(float *A, float *B, const int m, const int n);
float  svector_sum(float *v, int start, int n);
float  svector_norm(float *v, int start, int n);
float  svector_mean(float *v, int start, int n);
void   sreorder(float *A, int m, int n, int N);
void   smatmult(float *C, float *A, float *B, const int m, const int n, const int k, const char *TA, const char *TB);
void   smatmul(float *C, float *A, float *B, const int m, const int n, const int k);
void   smatmulp(float *C, float *A, float *B, const int m, const int n, const int k);
void   svecmat(float *v, float *A, const int m, const int n);
int    sinv(float *A, int m, int n);
int    sinverse(float *A, int N, char *UoL);
void   ssort(float *v, int *index, int n);
void   srandom_matrix(float *A, int m, int n, unsigned int seed);
void   seuclidean_d(float *D, float *X, const int m, const int n);
// Double version
void   dtranspose(double *A, double *B, const int m, const int n);
double dvector_sum(double *v, int start, int n);
double dvector_norm(double *v, int start, int n);
double dvector_mean(double *v, int start, int n);
void   dreorder(double *A, int m, int n, int N);
void   dmatmult(double *C, double *A, double *B, const int m, const int n, const int k, const char *TA, const char *TB);
void   dmatmul(double *C, double *A, double *B, const int m, const int n, const int k);
void   dmatmulp(double *C, double *A, double *B, const int m, const int n, const int k);
void   dvecmat(double *v, double *A, const int m, const int n);
int    dinv(double *A, int m, int n);
int    dinverse(double *A, int N, char *UoL);
void   dsort(double *v, int *index, int n);
void   drandom_matrix(double *A, int m, int n, unsigned int seed);
void   deuclidean_d(double *D, double *X, const int m, const int n);
// Float complex version
void   cmatmult(scomplex_t *C, scomplex_t *A, scomplex_t *B, const int m, const int n, const int k, const char *TA, const char *TB);
void   cmatmul(scomplex_t *C, scomplex_t *A, scomplex_t *B, const int m, const int n, const int k);
void   cmatmulp(scomplex_t *C, scomplex_t *A, scomplex_t *B, const int m, const int n, const int k);
void   cvecmat(scomplex_t *v, scomplex_t *A, const int m, const int n);
int    cinv(scomplex_t *A, int m, int n);
int    cinverse(scomplex_t *A, int N, char *UoL);
int    ceigen(float *real, float *imag, scomplex_t *vecs, float *A, const int m, const int n);
int    ccholesky(scomplex_t *A, int N);
void   cvandermonde(scomplex_t *Vand, float *real, float *imag, int m, int n);
void   cvandermondeTime(scomplex_t *Vand, float *real, float *imag, int m, int n, float *t);
void   csort(scomplex_t *v, int *index, int n);
// Double complex version
void   zmatmult(dcomplex_t *C, dcomplex_t *A, dcomplex_t *B, const int m, const int n, const int k, const char *TA, const char *TB);
void   zmatmul(dcomplex_t *C, dcomplex_t *A, dcomplex_t *B, const int m, const int n, const int k);
void   zmatmulp(dcomplex_t *C, dcomplex_t *A, dcomplex_t *B, const int m, const int n, const int k);
void   zvecmat(dcomplex_t *v, dcomplex_t *A, const int m, const int n);
int    zinv(dcomplex_t *A, int m, int n);
int    zinverse(dcomplex_t *A, int N, char *UoL);
int    zeigen(double *real, double *imag, dcomplex_t *vecs, double *A, const int m, const int n);
int    zcholesky(dcomplex_t *A, int N);
void   zvandermonde(dcomplex_t *Vand, double *real, double *imag, int m, int n);
void   zvandermondeTime(dcomplex_t *Vand, double *real, double *imag, int m, int n, double *t);
void   zsort(dcomplex_t *v, int *index, int n);
