/*
	Vector and matrix math operations
*/

#include <math.h>

#ifdef USE_MKL
#include "mkl.h"
#else
#include "cblas.h"
#endif

#include "vector_matrix.h"

#define BLK_LIM         5000
#define AC_MAT(A,n,i,j) *((A)+(n)*(i)+(j)) 
#define POW2(x)         ((x)*(x))


void transpose(double *A, const int m, const int n) {
	/*
		Naive approximation to matrix transpose.
		Overwrites A matrix
	*/
	double swp;
	for (int ii=0; ii<m; ++ii) {
		for (int jj=0; jj<ii+1; ++jj) {
			swp = AC_MAT(A,n,ii,jj);
			AC_MAT(A,n,ii,jj) = AC_MAT(A,n,jj,ii);
			AC_MAT(A,n,jj,ii) = swp;
		}
	}
}

double vector_norm(double *v, int start, int n) {
	/*
		Compute the norm of the n-dim vector v from the position start
	*/
	double norm = 0;
	#ifdef USE_OMP
	#pragma omp parallel for reduction(+:norm) shared(v) firstprivate(start,n)
	#endif
	for(int ii = start; ii < n; ++ii)
		norm += POW2(v[ii]);
	return sqrt(norm);
}

void reorder(double *A, int m, int n, int N) {
	/*
		Function which reorders the matrix A(m,n) to a matrix A(m,N)
		in order to delete the values that do not belong to the first N columns.
		
		Memory has to be reallocated after using the function.
	*/
	int ii = 0;
	for(int im = 0; im < m; ++im){
		for(int in = 0; in < N; ++in){
			A[ii] = AC_MAT(A,n,im,in);
			++ii;
		}
	}
}

void matmul(double *C, double *A, double *B, const int m, const int n, const int k) {
	/*
		Matrix multiplication C = A x B
		using cblas routines.

		C(m,n), A(m,k), B(k,n)
	*/
	cblas_dgemm(
		CblasRowMajor, // const CBLAS_LAYOUT 	  layout
		 CblasNoTrans, // const CBLAS_TRANSPOSE   TransA
		 CblasNoTrans, // const CBLAS_TRANSPOSE   TransB
		            m, // const CBLAS_INDEX 	  M
		            n, // const CBLAS_INDEX 	  N
		            k, // const CBLAS_INDEX 	  K
		          1.0, // const double 	          alpha
		            A, // const double * 	      A
		            k, // const CBLAS_INDEX 	  lda
	  			    B, // const double * 	      B
		            n, // const CBLAS_INDEX 	  ldb
		           0., // const double 	          beta
 				    C, // double * 	              C
		            n  // const CBLAS_INDEX 	  ldc
	);	
}

void vecmat(double *v, double *A, const int m, const int n) {
	/*
		Computes the product of b x A
		using cblas routines.

		A(m,n), b(m)
	*/
	#ifdef USE_OMP
	#pragma omp parallel for shared(b,A) firstprivate(m,n)
	#endif
	for(int ii=0; ii<m; ++ii)
		cblas_dscal(n,v[ii],A+n*ii,1);	
}
