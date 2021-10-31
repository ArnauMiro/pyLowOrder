/*
	SVD - Singular Value Decomposition of a matrix
*/
#define USE_LAPACK_DGESVD
#include <stdlib.h>
#include <string.h>
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
#include "svd.h"

#define AC_MAT(A,n,i,j) *((A)+(n)*(i)+(j))
#define MIN(a,b)         ((a)<(b)) ? (a) : (b)
#define MAX(a,b)         ((a)>(b)) ? (a) : (b)
#define POW2(x)          ((x)*(x))


int svd(double *U, double *S, double *VT, double *Y, const int m, const int n) {
	/*
		Single value decomposition (SVD) using Lapack.

		U(m,n)   are the POD modes and must come preallocated.
		S(n)     are the singular values.
		VT(n,n)  are the right singular vectors (transposed).

		Lapack dgesvd:
			http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_ga84fdf22a62b12ff364621e4713ce02f2.html
			https://www.netlib.org/lapack/explore-html/d0/dee/lapacke__dgesvd_8c_af31b3cb47f7cc3b9f6541303a2968c9f.html
		Lapack dgesdd (more optimized):
			http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_gad8e0f1c83a78d3d4858eaaa88a1c5ab1.html
			http://www.netlib.org/lapack//explore-html/d3/d23/lapacke__dgesdd_8c_aaf227f107a19ae6021f591c4de5fdbd5.html
		On ROW/COL major:
			https://stackoverflow.com/questions/34698550/understanding-lapack-row-major-and-lapack-col-major-with-lda
			https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/lapacke_dgesvd_row.c.htm
	*/
	int retval, mn = MIN(m,n);
	#ifdef USE_LAPACK_DGESVD
	// Run LAPACKE DGESVD for the single value decomposition
	double *superb;
	superb = (double*)malloc((mn-1)*sizeof(double));
	retval = LAPACKE_dgesvd(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
					 'S', // char  		jobu
					 'S', // char  		jobvt
					   m, // int  		m
					   n, // int  		n
					   Y, // double*  	a
					   n, // int  		lda
					   S, // double *  	s
					   U, // double *  	u
					  mn, // int  		ldu
					  VT, // double *  	vt
					   n, // int  		ldvt
				  superb  // double *  	superb
	);
	free(superb);
	#else
	// Run LAPACKE DGESDD for the single value decomposition
	retval = LAPACKE_dgesdd(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
					 'S', // char  		jobz
					   m, // int  		m
					   n, // int  		n
					   Y, // double*  	a
					   n, // int  		lda
					   S, // double *  	s
					   U, // double *  	u
					  mn, // int  		ldu
					  VT, // double *  	vt
					   n  // int  		ldvt
	);
	#endif
	return retval;
}


int tsqr_svd(double *Ui, double *S, double *VT, double *Ai, const int m, const int n, MPI_Comm comm) {
	/*
		Single value decomposition (SVD) using TSQR algorithm from
		T. Sayadi and P. J. Schmid, ‘Parallel data-driven decomposition algorithm 
		for large-scale datasets: with application to transitional boundary layers’, 
		Theor. Comput. Fluid Dyn., vol. 30, no. 5, pp. 415–428, Oct. 2016
		
		doi: 10.1007/s00162-016-0385-x

		Ai(m,n)  data matrix dispersed on each processor.

		Ui(m,n)  POD modes dispersed on each processor (must come preallocated).
		S(n)     singular values.
		VT(n,n)  right singular vectors (transposed).
	*/	
	int info = 0, mn = MIN(m,n), ii, jj;
	int mpi_rank, mpi_size;
	// Recover rank and size
	MPI_Comm_rank(comm,&mpi_rank);
	MPI_Comm_size(comm,&mpi_size);
	// Algorithm 1 from Sayadi and Schmid (2016) - Q and R matrices
	// Allocate memory
	double *Atmp, *tau, *R, *Rtmp, *Rp, *Qi;
	Atmp = (double*)malloc(m*n*sizeof(double));
	tau  = (double*)malloc(mn*sizeof(double));
	R    = (double*)malloc(n*n*sizeof(double));
	Rtmp = (double*)malloc(n*n*sizeof(double));
	Rp   = (double*)malloc(mpi_size*n*n*sizeof(double));
	Qi   = (double*)malloc(m*n*sizeof(double));
	// Copy A to Atmp
	memcpy(Atmp,Ai,m*n*sizeof(double));
	// Run LAPACK dgerqf - QR factorization on A
	info = LAPACKE_dgeqrf(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
					   m, // int  		m
					   n, // int  		n
					Atmp, // double*  	a
					   n, // int  		lda
					 tau  // double * 	tau 
	);
	if (!(info==0)) return info;
	// Copy Ri matrix
	memset(R,0,n*n*sizeof(double));
	for(ii=0;ii<n;++ii)
		for(jj=ii;jj<n;++jj)
			AC_MAT(R,n,ii,jj) = AC_MAT(Atmp,n,ii,jj);
	// Run LAPACK dorgqr - Generate Q matrix
	info = LAPACKE_dorgqr(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
					   m, // int  		m
					   n, // int  		n		
					   n, // int  		k
					Atmp, // double*  	a
					   n, // int  		lda					   		
					 tau  // double * 	tau 
	);
	if (!(info==0)) return info;
	// MPI_ALLGATHER to obtain R
	MPI_Allgather(R,n*n,MPI_DOUBLE,Rp,n*n,MPI_DOUBLE,comm);
	// Run LAPACK dgerqf - QR factorization on Rp
	info = LAPACKE_dgeqrf(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
			  mpi_size*n, // int  		m
					   n, // int  		n
					  Rp, // double*  	a
					   n, // int  		lda
					 tau  // double * 	tau 
	);
	if (!(info==0)) return info;
	// Copy R matrix - reusing R matrix
	memset(R,0,n*n*sizeof(double));
	for(ii=0;ii<n;++ii)
		for(jj=ii;jj<n;++jj)
			AC_MAT(R,n,ii,jj) = AC_MAT(Rp,n,ii,jj);
	// Run LAPACK dorgqr - Generate Q2 matrix
	info = LAPACKE_dorgqr(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
			  mpi_size*n, // int  		m
					   n, // int  		n		
					   n, // int  		k
					  Rp, // double*  	a
					   n, // int  		lda					   		
					 tau  // double * 	tau 
	);
	if (!(info==0)) return info;
	for(ii=0;ii<n;++ii)
		for(jj=0;jj<n;++jj)
			AC_MAT(Rtmp,n,ii,jj) = AC_MAT(Rp,n,ii+mpi_rank*n,jj);
	// Finally compute Qi = Atmp x Rp
	matmul(Qi,Atmp,Rtmp,m,n,n);
	// Free memory
	free(Atmp); free(tau); free(Rp); free(Rtmp);
	// At this point we have R and Qi scattered on the processors
	// Algorithm 2 from Sayadi and Schmid (2016) - Ui, S and VT
	double *Ur;
	Ur = (double*)malloc(n*n*sizeof(double));
	// Call SVD routine
	info = svd(Ur,S,VT,R,n,n);
	if (!(info==0)) return info;
	// Compute Ui = Qi x Ur
	matmul(Ui,Qi,Ur,m,n,n);
	// Free memory
	free(Ur); free(R); free(Qi);
	return info;
}
