/*
	Vector and matrix math operations
*/
#include <math.h>
#include <complex.h>
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

struct array_index
{
    double value;
    int index;
};


void transpose(double *A, double *B, const int m, const int n) {
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

double vector_norm(double *v, int start, int n) {
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

void reorder(double *A, int m, int n, int N) {
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

void matmul_paral(double *C, double *A, double *B, const int m, const int n, const int k) {
	/*
		Matrix multiplication C = A x B
		using cblas routines.

		C(m,n), A(m,k), B(k,n)
	*/
  double *Cmine;
  Cmine = (double*)malloc(m*n*sizeof(double));
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
 				    Cmine, // double * 	              C
		            n  // const CBLAS_INDEX 	  ldc
	);

  MPI_Allreduce(Cmine, C, m*n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  free(Cmine);
}

void matmul_complex(complex_t *C, complex_t *A, complex_t *B, const int m, const int n, const int k, char *TransA, char *TransB) {
	/*
		Complex matrix multiplication C = A x B
		using cblas routines.

		C(m,n), A(m,k), B(k,n)
		TRANSA: 0 - NoTrans; 1 - Trans; 2 - Hermitian
	*/
	complex_t alpha = 1 + 0*I;
	complex_t beta  = 0 + 0*I;
	CBLAS_TRANSPOSE transa;
	CBLAS_TRANSPOSE transb;
	CBLAS_INDEX lda;
	CBLAS_INDEX ldb;
	CBLAS_INDEX ldc;
	if(*TransA == 'N'){transa = CblasNoTrans;   lda = k; ldc = n;}
	if(*TransA == 'T'){transa = CblasTrans;     lda = m; ldc = n;}
	if(*TransA == 'C'){transa = CblasConjTrans; lda = m; ldc = n;}
	if(*TransB == 'N'){transb = CblasNoTrans;   ldb = n; ldc = n;}
	if(*TransB == 'T'){transb = CblasTrans;     ldb = k; ldc = n;}
	if(*TransB == 'C'){transb = CblasConjTrans; ldb = k; ldc = n;}
	cblas_zgemm(
		CblasRowMajor, // const CBLAS_LAYOUT 	  layout
		      transa, // const CBLAS_TRANSPOSE   TransA
		      transb, // const CBLAS_TRANSPOSE   TransB
		            m, // const CBLAS_INDEX 	  M
		            n, // const CBLAS_INDEX 	  N
		            k, // const CBLAS_INDEX 	  K
		       &alpha, // const double 	          alpha
		            A, // const complex_t * 	      A
		          lda, // const CBLAS_INDEX 	  lda
	  			      B, // const complex_t * 	      B
		          ldb, // const CBLAS_INDEX 	  ldb
		        &beta, // const double 	          beta
 				        C, // complex_t * 	              C
		          ldc  // const CBLAS_INDEX 	  ldc
	);
}

void vecmat(double *v, double *A, const int m, const int n) {
	/*
		Computes the product of b x A
		using cblas routines.

		A(m,n), b(m)
	*/
	int ii;
	#ifdef USE_OMP
	#pragma omp parallel for private(ii) shared(b,A) firstprivate(m,n)
	#endif
	for(ii=0; ii<m; ++ii) {
		cblas_dscal(n,v[ii],A+n*ii,1);
	}
}

void vecmat_complex(complex_t *v, complex_t *A, const int m, const int n) {
	/*
		Computes the product of b x A
		using cblas routines.

		A(m,n), b(m)
	*/
	int ii;
	#ifdef USE_OMP
	#pragma omp parallel for private(ii) shared(b,A) firstprivate(m,n)
	#endif
	for(ii=0; ii<m; ++ii) {
		cblas_zscal(n,&v[ii],A+n*ii,1);
	}
}

int eigen(double *real, double *imag, complex_t *w, double *A,
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
	double *vl;
	vl = (double*)malloc(n*n*sizeof(double));
	double *vecs;
  double tol = 1e-12;
	vecs = (double*)malloc(n*n*sizeof(double));
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

double RMSE(double *A, double *B, const int m, const int n, MPI_Comm comm) {
	/*
		Compute the Root Meean Square Error (RMSE) between two
		matrices and return it

		A(m,n), B(m,n)
	*/
	int ii, jj;
	double sum1 = 0., norm1 = 0., sum1g = 0.;
	double sum2 = 0., norm2 = 0., sum2g = 0.;
	#ifdef USE_OMP
	#pragma omp parallel for private(ii,jj) shared(A,B) firstprivate(m,n)
	#endif
	for(ii = 0; ii < n; ++ii) {
		norm1 = 0.;
		norm2 = 0.;
		for(jj = 0; jj < m; ++jj){
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

int cholesky(complex_t *A, int N){
	/*
	Compute the lower Cholesky factorization of A
	*/
	int info, ii, jj;
	info = LAPACKE_zpotrf(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
		'L', //char			Decide if the Upper or the Lower triangle of A are stored
		  N, //int			Order of matrix A
			A, //complex	Matrix A to decompose (works as input and output)
		  N //int			Leading dimension of A
	);
	for(ii = 0; ii < N; ++ii){
		for(jj = ii+1; jj < N; ++jj){
			AC_MAT(A,N,ii,jj) = 0 + 0*I;
		}
	}
	return info;
}

void vandermonde(complex_t *Vand, double *real, double *imag, int m, int n){
	/*
	Computes the Vandermonde matrix of a complex vector formed by real + imag*I
	*/
  int ii, jj;
	for(ii = 0; ii < m; ++ii){
		for(jj = 0; jj < n; ++jj){
			AC_MAT(Vand, n, ii, jj) = cpow((real[ii] + imag[ii]*I), jj);
		}
	}
}

int inverse(complex_t *A, int N, int UoL){
	/*
	Compute the lower Cholesky factorization of A
	*/
	int info;
	if(UoL == 0){
		info = LAPACKE_ztrtri(
			LAPACK_ROW_MAJOR, // int  		matrix_layout
		 	'U', //char			Decide if the Upper or the Lower triangle of A are stored
			'N', //int			Decide if is non Unitary or Unitary A
		  	N, //int			Order of A
				A, //complex	Matrix A to decompose (works as input and output)
				N //int			Leading dimension of A
			);
	}
	else{
		info = LAPACKE_ztrtri(
			LAPACK_ROW_MAJOR, // int  		matrix_layout
		 	'L', //char			Decide if the Upper or the Lower triangle of A are stored
			'N', //int			Decide if is non Unitary or Unitary A
		  	N, //int			Order of A
				A, //complex	Matrix A to decompose (works as input and output)
				N //int			Leading dimension of A
			);
	}
	return info;
}

int cmp(const void *a, const void *b)
{
    struct array_index *a1 = (struct array_index *)a;
    struct array_index *a2 = (struct array_index *)b;
    if ((*a1).value > (*a2).value)
        return -1;
    else if ((*a1).value < (*a2).value)
        return 1;
    else
        return 0;
}

void index_sort(double *v, int *index, int n){
    int i;
		struct array_index *objects = malloc(n*sizeof(struct array_index));
		for (i = 0; i < n; ++i){
        objects[i].value = v[i];
        objects[i].index = i;
    }
		qsort(objects, n, sizeof(objects[0]), cmp);
		for(i = 0; i < n; ++i){
			v[i]     = objects[i].value;
			index[i] = objects[i].index;
		}
		free(objects);
}
