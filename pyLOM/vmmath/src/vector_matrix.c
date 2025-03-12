/*
	Vector and matrix math operations
*/
#include <math.h>
#include <complex.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"
typedef float  _Complex scomplex_t;
typedef double _Complex dcomplex_t;

#ifdef USE_MKL
#define MKL_Complex8  scomplex_t
#define MKL_Complex16 dcomplex_t
#include "mkl.h"
#include "mkl_lapacke.h"
#else
#include "cblas.h"
#include "lapacke.h"
#endif

#include "vector_matrix.h"

#define BLK_LIM         5000
#define AC_MAT(A,n,i,j) *((A)+(n)*(i)+(j))
#define MIN(a,b)        ((a)<(b)) ? (a) : (b)
#define POW2(x)         ((x)*(x))


void stranspose(float *A, float *B, const int m, const int n) {
	/*
		Naive approximation to matrix transpose.
	*/
	int ii, jj;
	for (ii=0; ii<m; ++ii) {
		for (jj=0; jj<n; ++jj) {
			AC_MAT(B,m,jj,ii) = AC_MAT(A,n,ii,jj);
		}
	}
}

void dtranspose(double *A, double *B, const int m, const int n) {
	/*
		Naive approximation to matrix transpose.
	*/
	int ii, jj;
	for (ii=0; ii<m; ++ii) {
		for (jj=0; jj<n; ++jj) {
			AC_MAT(B,m,jj,ii) = AC_MAT(A,n,ii,jj);
		}
	}
}

float svector_sum(float *v, int start, int n) {
	/*
		Compute the sum of the n-dim vector v from the position start
	*/
	int ii;
	float sum = 0;
	#ifdef USE_OMP
	#pragma omp parallel for reduction(+:sum) private(ii) shared(v) firstprivate(start,n)
	#endif
	for(ii = start; ii < n; ++ii)
		sum += v[ii];
	return sum;
}

double dvector_sum(double *v, int start, int n) {
	/*
		Compute the sum of the n-dim vector v from the position start
	*/
	int ii;
	double sum = 0;
	#ifdef USE_OMP
	#pragma omp parallel for reduction(+:sum) private(ii) shared(v) firstprivate(start,n)
	#endif
	for(ii = start; ii < n; ++ii)
		sum += v[ii];
	return sum;
}

float svector_norm(float *v, int start, int n) {
	/*
		Compute the norm of the n-dim vector v from the position start
	*/
	int ii;
	float norm = 0;
	#ifdef USE_OMP
	#pragma omp parallel for reduction(+:norm) private(ii) shared(v) firstprivate(start,n)
	#endif
	for(ii = start; ii < n; ++ii)
		norm += POW2(v[ii]);
	return sqrt(norm);
}

double dvector_norm(double *v, int start, int n) {
	/*
		Compute the norm of the n-dim vector v from the position start
	*/
	int ii;
	double norm = 0;
	#ifdef USE_OMP
	#pragma omp parallel for reduction(+:norm) private(ii) shared(v) firstprivate(start,n)
	#endif
	for(ii = start; ii < n; ++ii)
		norm += POW2(v[ii]);
	return sqrt(norm);
}

float svector_mean(float *v, int start, int n) {
	/*
		Compute the mean of the n-dim vector v from the position start
	*/
	float sum = svector_sum(v,start,n);
	return sum/(float)(n);
}

double dvector_mean(double *v, int start, int n) {
	/*
		Compute the mean of the n-dim vector v from the position start
	*/
	double sum = dvector_sum(v,start,n);
	return sum/(double)(n);
}

void sreorder(float *A, int m, int n, int N) {
	/*
		Function which reorders the matrix A(m,n) to a matrix A(m,N)
		in order to delete the values that do not belong to the first N columns.

		Memory has to be reallocated after using the function.
	*/
	int ii = 0, im, in;
	for(im = 0; im < m; ++im){
		for(in = 0; in < N; ++in){
			A[ii] = AC_MAT(A,n,im,in);
			++ii;
		}
	}
}

void dreorder(double *A, int m, int n, int N) {
	/*
		Function which reorders the matrix A(m,n) to a matrix A(m,N)
		in order to delete the values that do not belong to the first N columns.

		Memory has to be reallocated after using the function.
	*/
	int ii = 0, im, in;
	for(im = 0; im < m; ++im){
		for(in = 0; in < N; ++in){
			A[ii] = AC_MAT(A,n,im,in);
			++ii;
		}
	}
}

void smatmult(float *C, float *A, float *B, const int m, const int n, const int k, const char *TA, const char *TB) {
	/*
		Matrix multiplication C = A x B
		using cblas routines.

		Transposable version

		C(m,n), A(m,k), B(k,n)
	*/
	float alpha = 1.0, beta = 0.0;
	CBLAS_TRANSPOSE TransA = CblasNoTrans, TransB = CblasNoTrans;
	CBLAS_INDEX     lda = k, ldb = n, ldc = n;
	// Transpose options
	if (*TA == 'T') {TransA = CblasTrans; lda = m;}
	if (*TB == 'T') {TransB = CblasTrans; ldb = k;}
	cblas_sgemm(
		CblasRowMajor, // const CBLAS_LAYOUT 	  layout
		       TransA, // const CBLAS_TRANSPOSE   TransA
		       TransB, // const CBLAS_TRANSPOSE   TransB
		            m, // const CBLAS_INDEX 	  M
			    n, // const CBLAS_INDEX 	  N
			    k, // const CBLAS_INDEX 	  K
			alpha, // const float 	          alpha
			    A, // const float * 	      A
			  lda, // const CBLAS_INDEX 	  lda
			    B, // const float * 	      B
			  ldb, // const CBLAS_INDEX 	  ldb
			 beta, // const float 	          beta
			    C, // float * 	              C
			  ldc  // const CBLAS_INDEX 	  ldc
	);
}

void dmatmult(double *C, double *A, double *B, const int m, const int n, const int k, const char *TA, const char *TB) {
	/*
		Matrix multiplication C = A x B
		using cblas routines.

		Transposable version

		C(m,n), A(m,k), B(k,n)
	*/
	double alpha = 1.0, beta = 0.0;
	CBLAS_TRANSPOSE TransA = CblasNoTrans, TransB = CblasNoTrans;
	CBLAS_INDEX     lda = k, ldb = n, ldc = n;
	// Transpose options
	if (*TA == 'T') {TransA = CblasTrans; lda = m;}
	if (*TB == 'T') {TransB = CblasTrans; ldb = k;}
	cblas_dgemm(
		CblasRowMajor, // const CBLAS_LAYOUT 	  layout
		       TransA, // const CBLAS_TRANSPOSE   TransA
		       TransB, // const CBLAS_TRANSPOSE   TransB
		            m, // const CBLAS_INDEX 	  M
			    n, // const CBLAS_INDEX 	  N
			    k, // const CBLAS_INDEX 	  K
			alpha, // const double 	          alpha
			    A, // const double * 	      A
			  lda, // const CBLAS_INDEX 	  lda
			    B, // const double * 	      B
			  ldb, // const CBLAS_INDEX 	  ldb
			 beta, // const double 	          beta
			    C, // double * 	              C
			  ldc  // const CBLAS_INDEX 	  ldc
	);
}

void smatmul(float *C, float *A, float *B, const int m, const int n, const int k) {
	/*
		Matrix multiplication C = A x B
		using cblas routines.

		C(m,n), A(m,k), B(k,n)
	*/
	smatmult(C,A,B,m,n,k,"N","N");
}

void dmatmul(double *C, double *A, double *B, const int m, const int n, const int k) {
	/*
		Matrix multiplication C = A x B
		using cblas routines.

		C(m,n), A(m,k), B(k,n)
	*/
	dmatmult(C,A,B,m,n,k,"N","N");
}

void cmatmult(scomplex_t *C, scomplex_t *A, scomplex_t *B, const int m, const int n, const int k, const char *TA, const char *TB) {
	/*
		Matrix multiplication C = A x B
		using cblas routines.

		C(m,n), A(m,k), B(k,n)
	*/
	scomplex_t alpha = 1.0 + 0.0*I, beta = 0.0 + 0.0*I;
	CBLAS_TRANSPOSE TransA = CblasNoTrans, TransB = CblasNoTrans;
	CBLAS_INDEX     lda = k, ldb = n, ldc = n;
	// Transpose options
	if (*TA == 'T'){ TransA = CblasTrans;     lda = m; }
	if (*TA == 'C'){ TransA = CblasConjTrans; lda = m; }
	if (*TB == 'T'){ TransB = CblasTrans;     ldb = k; }
	if (*TB == 'C'){ TransB = CblasConjTrans; ldb = k; }
	cblas_cgemm(
		CblasRowMajor, // const CBLAS_LAYOUT 	  layout
		       TransA, // const CBLAS_TRANSPOSE   TransA
		       TransB, // const CBLAS_TRANSPOSE   TransB
			    m, // const CBLAS_INDEX 	  M
			    n, // const CBLAS_INDEX 	  N
			    k, // const CBLAS_INDEX 	  K
		       &alpha, // const scomplex_t 	      alpha
			    A, // const scomplex_t * 	  A
			  lda, // const CBLAS_INDEX 	  lda
			    B, // const scomplex_t * 	  B
			  ldb, // const CBLAS_INDEX 	  ldb
			&beta, // const scomplex_t 	      beta
			    C, // scomplex_t * 	          C
			  ldc  // const CBLAS_INDEX 	  ldc
	);
}

void zmatmult(dcomplex_t *C, dcomplex_t *A, dcomplex_t *B, const int m, const int n, const int k, const char *TA, const char *TB) {
	/*
		Matrix multiplication C = A x B
		using cblas routines.

		C(m,n), A(m,k), B(k,n)
	*/
	dcomplex_t alpha = 1.0 + 0.0*I, beta = 0.0 + 0.0*I;
	CBLAS_TRANSPOSE TransA = CblasNoTrans, TransB = CblasNoTrans;
	CBLAS_INDEX     lda = k, ldb = n, ldc = n;
	// Transpose options
	if (*TA == 'T'){ TransA = CblasTrans;     lda = m; }
	if (*TA == 'C'){ TransA = CblasConjTrans; lda = m; }
	if (*TB == 'T'){ TransB = CblasTrans;     ldb = k; }
	if (*TB == 'C'){ TransB = CblasConjTrans; ldb = k; }
	cblas_zgemm(
		CblasRowMajor, // const CBLAS_LAYOUT 	  layout
		       TransA, // const CBLAS_TRANSPOSE   TransA
		       TransB, // const CBLAS_TRANSPOSE   TransB
			    m, // const CBLAS_INDEX 	  M
			    n, // const CBLAS_INDEX 	  N
			    k, // const CBLAS_INDEX 	  K
		       &alpha, // const dcomplex_t 	      alpha
			    A, // const dcomplex_t * 	  A
			  lda, // const CBLAS_INDEX 	  lda
			    B, // const dcomplex_t * 	  B
			  ldb, // const CBLAS_INDEX 	  ldb
			&beta, // const dcomplex_t 	      beta
			    C, // dcomplex_t * 	          C
			  ldc  // const CBLAS_INDEX 	  ldc
	);
}

void cmatmul(scomplex_t *C, scomplex_t *A, scomplex_t *B, const int m, const int n, const int k) {
	/*
		Matrix multiplication C = A x B
		using cblas routines.

		C(m,n), A(m,k), B(k,n)
	*/
	cmatmult(C,A,B,m,n,k,"N","N");
}

void zmatmul(dcomplex_t *C, dcomplex_t *A, dcomplex_t *B, const int m, const int n, const int k) {
	/*
		Matrix multiplication C = A x B
		using cblas routines.

		C(m,n), A(m,k), B(k,n)
	*/
	zmatmult(C,A,B,m,n,k,"N","N");
}

void smatmulp(float *C, float *A, float *B, const int m, const int n, const int k) {
	/*
		Matrix multiplication C = A x B
		using cblas routines.

		C(m,n), A(m,k), B(k,n)
	*/
	float *Cmine;
	Cmine = (float*)malloc(m*n*sizeof(float));
	smatmul(Cmine,A,B,m,n,k);
	MPI_Allreduce(Cmine, C, m*n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
	free(Cmine);
}

void dmatmulp(double *C, double *A, double *B, const int m, const int n, const int k) {
	/*
		Matrix multiplication C = A x B
		using cblas routines.

		C(m,n), A(m,k), B(k,n)
	*/
	double *Cmine;
	Cmine = (double*)malloc(m*n*sizeof(double));
	dmatmul(Cmine,A,B,m,n,k);
	MPI_Allreduce(Cmine, C, m*n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	free(Cmine);
}

void cmatmulp(scomplex_t *C, scomplex_t *A, scomplex_t *B, const int m, const int n, const int k) {
	/*
		Matrix multiplication C = A x B
		using cblas routines.

		C(m,n), A(m,k), B(k,n)
	*/
	scomplex_t *Cmine;
	Cmine = (scomplex_t*)malloc(m*n*sizeof(scomplex_t));
	cmatmul(Cmine,A,B,m,n,k);
	MPI_Allreduce(Cmine, C, m*n, MPI_C_FLOAT_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
	free(Cmine);
}

void zmatmulp(dcomplex_t *C, dcomplex_t *A, dcomplex_t *B, const int m, const int n, const int k) {
	/*
		Matrix multiplication C = A x B
		using cblas routines.

		C(m,n), A(m,k), B(k,n)
	*/
	dcomplex_t *Cmine;
	Cmine = (dcomplex_t*)malloc(m*n*sizeof(dcomplex_t));
	zmatmul(Cmine,A,B,m,n,k);
	MPI_Allreduce(Cmine, C, m*n, MPI_C_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
	free(Cmine);
}

void svecmat(float *v, float *A, const int m, const int n) {
	/*
		Computes the product of b x A
		using cblas routines.

		A(m,n), b(m)
	*/
	int ii;
	#ifdef USE_OMP
	#pragma omp parallel for private(ii) shared(b,A) firstprivate(m,n)
	#endif
	for(ii=0; ii<m; ++ii)
		cblas_sscal(n,v[ii],A+n*ii,1);
}

void dvecmat(double *v, double *A, const int m, const int n) {
	/*
		Computes the product of b x A
		using cblas routines.

		A(m,n), b(m)
	*/
	int ii;
	#ifdef USE_OMP
	#pragma omp parallel for private(ii) shared(b,A) firstprivate(m,n)
	#endif
	for(ii=0; ii<m; ++ii)
		cblas_dscal(n,v[ii],A+n*ii,1);
}

void cvecmat(scomplex_t *v, scomplex_t *A, const int m, const int n) {
	/*
		Computes the product of b x A
		using cblas routines.

		A(m,n), b(m)
	*/
	int ii;
	#ifdef USE_OMP
	#pragma omp parallel for private(ii) shared(b,A) firstprivate(m,n)
	#endif
	for(ii=0; ii<m; ++ii)
		cblas_cscal(n,&v[ii],A+n*ii,1);
}

void zvecmat(dcomplex_t *v, dcomplex_t *A, const int m, const int n) {
	/*
		Computes the product of b x A
		using cblas routines.

		A(m,n), b(m)
	*/
	int ii;
	#ifdef USE_OMP
	#pragma omp parallel for private(ii) shared(b,A) firstprivate(m,n)
	#endif
	for(ii=0; ii<m; ++ii)
		cblas_zscal(n,&v[ii],A+n*ii,1);
}

int ceigen(float *real, float *imag, scomplex_t *w, float *A,
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
	int info, ivec, imod;
	float *vl, *vecs;
	float tol = 1e-12;
	vl   = (float*)malloc(n*n*sizeof(float));
	vecs = (float*)malloc(n*n*sizeof(float));
	info = LAPACKE_sgeev(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
			     'N', // char       jobvl
			     'V', // char       jobvr
			       n, // int        n
			       A, // float*    A
			       m, // int        lda
			    real, // float*    wr
			    imag, // float*    wi
			      vl, // float*    vl
			       n, // int        ldvl
			    vecs, // float*    vr
			       n  // int        ldvr
	);
	//Define and allocate memory for the complex array of eigenvectors
	//Change while for a for
	for (imod = 0; imod < n; imod++){
		if (imag[imod] > tol){//If the imaginary part is greater than zero, the eigenmode has a conjugate.
			for (ivec = 0; ivec < n; ivec++){
				AC_MAT(w,n,ivec,imod)   = AC_MAT(vecs,n,ivec,imod) + AC_MAT(vecs,n,ivec,imod+1)*I;
				AC_MAT(w,n,ivec,imod+1) = AC_MAT(vecs,n,ivec,imod) - AC_MAT(vecs,n,ivec,imod+1)*I;
			}
			imod += 1;
		}
		else{
			for (ivec = 0; ivec < n; ivec++){
				AC_MAT(w,n,ivec,imod)   = AC_MAT(vecs,n,ivec,imod) + 0*I;
			}
		}
	}
	free(vecs);
	free(vl);
	return info;
}

int zeigen(double *real, double *imag, dcomplex_t *w, double *A,
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
	int info, ivec, imod;
	double *vl, *vecs;
	double tol = 1e-12;
	vl   = (double*)malloc(n*n*sizeof(double));
	vecs = (double*)malloc(n*n*sizeof(double));
	info = LAPACKE_dgeev(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
			     'N', // char       jobvl
			     'V', // char       jobvr
			       n, // int        n
			       A, // double*    A
			       m, // int        lda
			    real, // double*    wr
			    imag, // double*    wi
			      vl, // double*    vl
			       n, // int        ldvl
			    vecs, // double*    vr
			       n  // int        ldvr
	);
	//Define and allocate memory for the complex array of eigenvectors
	//Change while for a for
	for (imod = 0; imod < n; imod++){
		if (imag[imod] > tol){//If the imaginary part is greater than zero, the eigenmode has a conjugate.
			for (ivec = 0; ivec < n; ivec++){
				AC_MAT(w,n,ivec,imod)   = AC_MAT(vecs,n,ivec,imod) + AC_MAT(vecs,n,ivec,imod+1)*I;
				AC_MAT(w,n,ivec,imod+1) = AC_MAT(vecs,n,ivec,imod) - AC_MAT(vecs,n,ivec,imod+1)*I;
			}
			imod += 1;
		}
		else{
			for (ivec = 0; ivec < n; ivec++){
				AC_MAT(w,n,ivec,imod)   = AC_MAT(vecs,n,ivec,imod) + 0*I;
			}
		}
	}
	free(vecs);
	free(vl);
	return info;
}

int ccholesky(scomplex_t *A, int N){
	/*
		Compute the lower Cholesky factorization of A
	*/
	int info, ii, jj;
	info = LAPACKE_cpotrf(
		 LAPACK_ROW_MAJOR, // int  	matrix_layout
			      'L', //char	Decide if the Upper or the Lower triangle of A are stored
		 		N, //int		Order of matrix A
		 		A, //complex	Matrix A to decompose (works as input and output)
		 		N  //int		Leading dimension of A
	);
	// Zero upper size part
	#ifdef USE_OMP
	#pragma omp parallel for collapse(2) private(ii,jj) shared(A) firstprivate(N)
	#endif
	for(ii = 0; ii < N; ++ii)
		for(jj = ii+1; jj < N; ++jj)
			AC_MAT(A,N,ii,jj) = 0.0 + 0.0*I;
	return info;
}

int zcholesky(dcomplex_t *A, int N){
	/*
		Compute the lower Cholesky factorization of A
	*/
	int info, ii, jj;
	info = LAPACKE_zpotrf(
		 LAPACK_ROW_MAJOR, // int  	matrix_layout
			      'L', //char	Decide if the Upper or the Lower triangle of A are stored
				N, //int		Order of matrix A
				A, //complex	Matrix A to decompose (works as input and output)
				N  //int		Leading dimension of A
	);
	// Zero upper size part
	#ifdef USE_OMP
	#pragma omp parallel for collapse(2) private(ii,jj) shared(A) firstprivate(N)
	#endif
	for(ii = 0; ii < N; ++ii)
		for(jj = ii+1; jj < N; ++jj)
			AC_MAT(A,N,ii,jj) = 0.0 + 0.0*I;
	return info;
}

void cvandermonde(scomplex_t *Vand, float *real, float *imag, int m, int n){
	/*
		Computes the Vandermonde matrix of a complex vector formed by real + imag*I
	*/
	int ii, jj;
	#ifdef USE_OMP
	#pragma omp parallel for collapse(2) private(ii,jj) shared(Vand,real,imag) firstprivate(m,n)
	#endif
	for(ii = 0; ii < m; ++ii){
		for(jj = 0; jj < n; ++jj){
			AC_MAT(Vand, n, ii, jj) = cpow((real[ii] + imag[ii]*I), jj);
		}
	}
}

void zvandermonde(dcomplex_t *Vand, double *real, double *imag, int m, int n){
	/*
		Computes the Vandermonde matrix of a complex vector formed by real + imag*I
	*/
	int ii, jj;
	#ifdef USE_OMP
	#pragma omp parallel for collapse(2) private(ii,jj) shared(Vand,real,imag) firstprivate(m,n)
	#endif
	for(ii = 0; ii < m; ++ii){
		for(jj = 0; jj < n; ++jj){
			AC_MAT(Vand, n, ii, jj) = cpow((real[ii] + imag[ii]*I), jj);
		}
	}
}

void cvandermondeTime(scomplex_t *Vand, float *real, float *imag, int m, int n, float *t){
	/*
		Computes the Vandermonde matrix of a complex vector formed by real + imag*I
	*/
	int ii, jj;
 	#ifdef USE_OMP
	#pragma omp parallel for collapse(2) private(ii,jj) shared(Vand,real,imag,t) firstprivate(m,n)
	#endif
	for(ii = 0; ii < m; ++ii){
		for(jj = 0; jj < n; ++jj){
			AC_MAT(Vand, n, ii, jj) = cpow((real[ii] + imag[ii]*I), t[jj]);
		}
	}
}

void zvandermondeTime(dcomplex_t *Vand, double *real, double *imag, int m, int n, double *t){
	/*
		Computes the Vandermonde matrix of a complex vector formed by real + imag*I
	*/
	int ii, jj;
 	#ifdef USE_OMP
	#pragma omp parallel for collapse(2) private(ii,jj) shared(Vand,real,imag,t) firstprivate(m,n)
	#endif
	for(ii = 0; ii < m; ++ii){
		for(jj = 0; jj < n; ++jj){
			AC_MAT(Vand, n, ii, jj) = cpow((real[ii] + imag[ii]*I), t[jj]);
		}
	}
}

int sinv(float *A, int m, int n) {
	/*
		Compute the inverse of A
	*/
	int info, *ipiv, mn = MIN(m,n);
	ipiv = (int*)malloc(mn*sizeof(int));

	info = LAPACKE_sgetrf(
		LAPACK_ROW_MAJOR, // int    matrix_layout
		               m, // int 	The number of rows of the matrix A.
                       n, // int	The number of columns of the matrix A.
					   A, // A is FLOAT PRECISION array, dimension (LDA,N), On exit, the factors L and U from the factorization
					   m, // int	The leading dimension of the array A.
					ipiv  // IPIV is INTEGER array, dimension (min(M,N))
	);
	if (info < 0) return info;

	info = LAPACKE_sgetri(
		LAPACK_ROW_MAJOR, // int    matrix_layout
		               n, // int	The order of the matrix A.
					   A, //  A is FLOAT PRECISION array, dimension (LDA,N). On entry, the factors L and U from the factorization
					   m, // int	The leading dimension of the array A.
					ipiv  // IPIV is INTEGER array, dimension (min(M,N))
	);
	return info;
}

int dinv(double *A, int m, int n) {
	/*
		Compute the inverse of A
	*/
	int info, *ipiv, mn = MIN(m,n);
	ipiv = (int*)malloc(mn*sizeof(int));

	info = LAPACKE_dgetrf(
		LAPACK_ROW_MAJOR, // int    matrix_layout
		               m, // int 	The number of rows of the matrix A.
                       n, // int	The number of columns of the matrix A.
					   A, // A is DOUBLE PRECISION array, dimension (LDA,N), On exit, the factors L and U from the factorization
					   m, // int	The leading dimension of the array A.
					ipiv  // IPIV is INTEGER array, dimension (min(M,N))
	);
	if (info < 0) return info;

	info = LAPACKE_dgetri(
		LAPACK_ROW_MAJOR, // int    matrix_layout
		               n, // int	The order of the matrix A.
					   A, //  A is DOUBLE PRECISION array, dimension (LDA,N). On entry, the factors L and U from the factorization
					   m, // int	The leading dimension of the array A.
					ipiv  // IPIV is INTEGER array, dimension (min(M,N))
	);
	return info;
}

int cinv(scomplex_t *A, int m, int n) {
	/*
		Compute the inverse of A
	*/
	int info, *ipiv, mn = MIN(m,n);
	ipiv = (int*)malloc(mn*sizeof(int));

	info = LAPACKE_cgetrf(
		LAPACK_ROW_MAJOR, // int    matrix_layout
		               m, // int 	The number of rows of the matrix A.
                       n, // int	The number of columns of the matrix A.
					   A, // A is COMPLEX PRECISION array, dimension (LDA,N), On exit, the factors L and U from the factorization
					   m, // int	The leading dimension of the array A.
					ipiv  // IPIV is INTEGER array, dimension (min(M,N))
	);
	if (info < 0) return info;

	info = LAPACKE_cgetri(
		LAPACK_ROW_MAJOR, // int    matrix_layout
		               n, // int	The order of the matrix A.
					   A, //  A is COMPLEX PRECISION array, dimension (LDA,N). On entry, the factors L and U from the factorization
					   m, // int	The leading dimension of the array A.
					ipiv  // IPIV is INTEGER array, dimension (min(M,N))
	);
	return info;
}

int zinv(dcomplex_t *A, int m, int n) {
	/*
		Compute the inverse of A
	*/
	int info, *ipiv, mn = MIN(m,n);
	ipiv = (int*)malloc(mn*sizeof(int));

	info = LAPACKE_zgetrf(
		LAPACK_ROW_MAJOR, // int    matrix_layout
		               m, // int 	The number of rows of the matrix A.
                       n, // int	The number of columns of the matrix A.
					   A, // A is COMPLEX PRECISION array, dimension (LDA,N), On exit, the factors L and U from the factorization
					   m, // int	The leading dimension of the array A.
					ipiv  // IPIV is INTEGER array, dimension (min(M,N))
	);
	if (info < 0) return info;

	info = LAPACKE_zgetri(
		LAPACK_ROW_MAJOR, // int    matrix_layout
		               n, // int	The order of the matrix A.
					   A, //  A is COMPLEX PRECISION array, dimension (LDA,N). On entry, the factors L and U from the factorization
					   m, // int	The leading dimension of the array A.
					ipiv  // IPIV is INTEGER array, dimension (min(M,N))
	);
	return info;
}

int sinverse(float *A, int N, char *UoL){
	/*
		Compute the inverse of A
		A must be an upper or lower triangular matrix
	*/
	int info;
	info = LAPACKE_strtri(
		 LAPACK_ROW_MAJOR, //int     matrix_layout
			     *UoL, //char    Decide if the Upper or the Lower triangle of A are stored
			      'N', //int	    Decide if is non Unitary or Unitary A
				N, //int	    Order of A
				A, //double  Matrix A to decompose (works as input and output)
				N  //int     Leading dimension of A
	);
	return info;
}

int dinverse(double *A, int N, char *UoL){
	/*
		Compute the inverse of A
		A must be an upper or lower triangular matrix
	*/
	int info;
	info = LAPACKE_dtrtri(
		 LAPACK_ROW_MAJOR, //int     matrix_layout
			     *UoL, //char    Decide if the Upper or the Lower triangle of A are stored
			      'N', //int	    Decide if is non Unitary or Unitary A
				N, //int	    Order of A
				A, //double  Matrix A to decompose (works as input and output)
				N  //int     Leading dimension of A
	);
	return info;
}

int cinverse(scomplex_t *A, int N, char *UoL){
	/*
		Compute the inverse of A
		A must be an upper or lower triangular matrix
	*/
	int info;
	info = LAPACKE_ctrtri(
		 LAPACK_ROW_MAJOR, //int     matrix_layout
		 	     *UoL, //char    Decide if the Upper or the Lower triangle of A are stored
			      'N', //int	    Decide if is non Unitary or Unitary A
				N, //int	    Order of A
				A, //complex Matrix A to decompose (works as input and output)
				N  //int     Leading dimension of A
	);
	return info;
}

int zinverse(dcomplex_t *A, int N, char *UoL){
	/*
		Compute the inverse of A
		A must be an upper or lower triangular matrix
	*/
	int info;
	info = LAPACKE_ztrtri(
		LAPACK_ROW_MAJOR, //int     matrix_layout
			    *UoL, //char    Decide if the Upper or the Lower triangle of A are stored
			     'N', //int	    Decide if is non Unitary or Unitary A
			       N, //int	    Order of A
			       A, //complex Matrix A to decompose (works as input and output)
			       N  //int     Leading dimension of A
	);
	return info;
}

int scompare(const void* a, const void* b) {
	float c1 = *(float*)a;
	float c2 = *(float*)b;
	float diff = fabs(c1) - fabs(c2);
	if (diff > 0.) return 1;
	else if (diff < 0.) return -1;
	else return 0;
}

int dcompare(const void* a, const void* b) {
	double c1 = *(double*)a;
	double c2 = *(double*)b;
	double diff = fabs(c1) - fabs(c2);
	if (diff > 0.) return 1;
	else if (diff < 0.) return -1;
	else return 0;
}

int ccompare(const void* a, const void* b) {
	scomplex_t c1 = *(scomplex_t*)a;
	scomplex_t c2 = *(scomplex_t*)b;
	float diff = cabs(c1) - cabs(c2);
	if (diff > 0.) return 1;
	else if (diff < 0.) return -1;
	else return 0;
}

int zcompare(const void* a, const void* b) {
	dcomplex_t c1 = *(dcomplex_t*)a;
	dcomplex_t c2 = *(dcomplex_t*)b;
	double diff = cabs(c1) - cabs(c2);
	if (diff > 0.) return 1;
	else if (diff < 0.) return -1;
	else return 0;
}

void ssort(float *v, int *index, int n){
	/*
		Returns the ordered indexes of a complex array according to the 
		absolute value of its elements
	*/
	int ii, jj;
	float *w;
	w = (float*)malloc(n*sizeof(float));

	memcpy(w,v,n*sizeof(float));
	qsort(w,n,sizeof(float),scompare);
	
	for (ii = 0; ii < n; ++ii) {
		for (jj = 0; jj < n; ++jj) {
			if (v[ii] == w[jj]) { index[ii] = jj; break; }
		}
	}
	free(w);
}

void dsort(double *v, int *index, int n){
	/*
		Returns the ordered indexes of a complex array according to the 
		absolute value of its elements
	*/
	int ii, jj;
	double *w;
	w = (double*)malloc(n*sizeof(double));

	memcpy(w,v,n*sizeof(double));
	qsort(w,n,sizeof(double),dcompare);
	
	for (ii = 0; ii < n; ++ii) {
		for (jj = 0; jj < n; ++jj) {
			if (v[ii] == w[jj]) { index[ii] = jj; break; }
		}
	}
	free(w);
}

void csort(scomplex_t *v, int *index, int n){
	/*
		Returns the ordered indexes of a complex array according to the 
		absolute value of its elements
	*/
	int ii, jj;
	scomplex_t *w;
	w = (scomplex_t*)malloc(n*sizeof(scomplex_t));

	memcpy(w,v,n*sizeof(scomplex_t));
	qsort(w,n,sizeof(scomplex_t),ccompare);

	for (ii = 0; ii < n; ++ii) {
		for (jj = 0; jj < n; ++jj) {
			if ((creal(v[ii]) == creal(w[jj])) & (cimag(v[ii]) == cimag(w[jj]))) { index[ii] = jj; break; }
		}
	}
	free(w);
}

void zsort(dcomplex_t *v, int *index, int n){
	/*
		Returns the ordered indexes of a complex array according to the 
		absolute value of its elements
	*/
	int ii, jj;
	dcomplex_t *w;
	w = (dcomplex_t*)malloc(n*sizeof(dcomplex_t));

	memcpy(w,v,n*sizeof(dcomplex_t));
	qsort(w,n,sizeof(dcomplex_t),zcompare);

	for (ii = 0; ii < n; ++ii) {
		for (jj = 0; jj < n; ++jj) {
			if ((creal(v[ii]) == creal(w[jj])) & (cimag(v[ii]) == cimag(w[jj]))) { index[ii] = jj; break; }
		}
	}
	free(w);
}

void srandom_matrix(float *A, int m, int n, unsigned int seed){
	/*
		Generate a single precision random matrix
	*/
	// Seed the random number generator
	srand(seed);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			AC_MAT(A,n,i,j) = (float)(rand()) / (float)(RAND_MAX);
		}
	}
}

void drandom_matrix(double *A, int m, int n, unsigned int seed){
	/*
		Generate a double precision random matrix
	*/
	// Seed the random number generator
	srand(seed);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			AC_MAT(A,n,i,j) = (double)(rand()) / (double)(RAND_MAX);
		}
	}
}