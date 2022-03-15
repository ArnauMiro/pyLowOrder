/*
	Vector and matrix math operations
*/
#include <complex.h>
typedef double _Complex complex_t;

void   transpose(double *A, const int m, const int n);
double vector_norm(double *v, int start, int n);
void   reorder(double *A, int m, int n, int N);
void   matmul(double *C, double *A, double *B, const int m, const int n, const int k);
void   matmul_complex(complex_t *C, complex_t *A, complex_t *B, const int m, const int n, const int k);
void   vecmat(double *v, double *A, const int m, const int n);
int    eigen(double *real, double *imag, complex_t *vecs, double *A, const int m, const int n);
double RMSE(double *A, double *B, const int m, const int n, MPI_Comm comm);
int    cholesky(complex_t *A, int N);
