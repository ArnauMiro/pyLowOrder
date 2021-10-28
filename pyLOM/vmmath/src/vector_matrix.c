/*
	Vector and matrix math operations
*/
#include <math.h>
#include "mpi.h"

#ifdef USE_MKL
#include "mkl.h"
#include "mkl_lapacke.h"
#else
#include "cblas.h"
#include "lapacke.h"
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
	for(int ii=0; ii<m; ++ii) {
		cblas_dscal(n,v[ii],A+n*ii,1);
	}
}

int eigen(double *real, double *imag, double *vecs, double *A, 
	const int m, const int n) {
	/*
		Compute the eigenvalues and eigenvectors of a matrix A using
		LAPACK functions.

		All inputs should come preallocated.

		real(n)   real eigenvalue part
		imag(n)   imaginary eigenvalue part
		vecs(n,n) eigenvectors

		A(m,n)   matrix to obtain eigenvalues and eigenvectors from
	*/
	int info;
	double *vl; 
	vl = (double*)malloc(1*sizeof(double));
	info = LAPACKE_dgeev(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
		           'N',   // char       jobvl
		           'V',   // char       jobvr
		             n,   // int        n
		             A,   // double*    A
		             m,   // int        lda
		          real,   // double*    wr
		          imag,   // double*    wi
		            vl,   // double*    vl
		             n,   // int        ldvl
		          vecs,   // double*    vr
		             n    // int        ldvr
	);
	free(vl);
	return info;
}

double RMSE(double *A, double *B, const int m, const int n, MPI_Comm comm) {
	/*
		Compute the Root Meean Square Error (RMSE) between two
		matrices and return it

		A(m,n), B(m,n)
	*/
	double sum1 = 0., norm1 = 0., sum1g = 0.;
	double sum2 = 0., norm2 = 0., sum2g = 0.;
	#ifdef USE_OMP
	#pragma omp parallel for shared(A,B) firstprivate(m,n)
	#endif
	for(int ii = 0; ii < n; ++ii) {
		norm1 = 0.;
		norm2 = 0.;
		for(int jj = 0; jj < m; ++jj){
			norm1 += POW2(AC_MAT(A,n,ii,jj) - AC_MAT(B,n,ii,jj));
			norm2 += POW2(AC_MAT(A,n,ii,jj));
		}
		sum1 += norm1;
		sum2 += norm2;
	}
	// Reduce MPI parallel run
	MPI_Allreduce(&sum1,&sum1g,1,MPI_DOUBLE,MPI_SUM,comm); 
	MPI_Allreduce(&sum2,&sum2g,1,MPI_DOUBLE,MPI_SUM,comm);
	// Return
	return sqrt(sum1g/sum2g);
}
