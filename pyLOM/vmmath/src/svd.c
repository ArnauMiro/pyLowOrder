/*
	SVD - Singular Value Decomposition of a matrix
*/
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <stdio.h>
#include "mpi.h"
typedef float  _Complex scomplex_t;
typedef double _Complex dcomplex_t;

#ifdef USE_MKL
#define MKL_Complex8  scomplex_t
#define MKL_Complex16 dcomplex_t
#include "mkl.h"
#include "mkl_lapacke.h"
#else
// Due to a bug on OpenBLAS we need to switch the
// SVD computation routine
#define USE_LAPACK_GESVD
#include "cblas.h"
#include "lapacke.h"
#endif

#include "vector_matrix.h"
#include "svd.h"

#define AC_MAT(A,n,i,j) *((A)+(n)*(i)+(j))
#define MIN(a,b)         ((a)<(b)) ? (a) : (b)
#define MAX(a,b)         ((a)>(b)) ? (a) : (b)
#define POW2(x)          ((x)*(x))


int ssvd(float *U, float *S, float *VT, float *Y, const int m, const int n) {
	/*
		Single value decomposition (SVD) using Lapack.

		U(m,n)   are the POD modes and must come preallocated.
		S(n)     are the singular values.
		VT(n,n)  are the right singular vectors (transposed).
	*/
	int retval = 0, mn = MIN(m,n);
	#ifdef USE_LAPACK_GESVD
	// Run LAPACKE SGESVD for the single value decomposition
	float *superb;
	superb = (float*)malloc((mn-1)*sizeof(float));
	retval = LAPACKE_sgesvd(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
					 'S', // char  		jobu
					 'S', // char  		jobvt
					   m, // int  		m
					   n, // int  		n
					   Y, // float*  	a
					   n, // int  		lda
					   S, // float *  	s
					   U, // float *  	u
					  mn, // int  		ldu
					  VT, // float *  	vt
					   n, // int  		ldvt
				  superb  // float *  	superb
	);
	free(superb);
	#else
	// Run LAPACKE SGESDD for the single value decomposition
	retval = LAPACKE_sgesdd(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
					 'S', // char  		jobz
					   m, // int  		m
					   n, // int  		n
					   Y, // float*  	a
					   n, // int  		lda
					   S, // float *  	s
					   U, // float *  	u
					  mn, // int  		ldu
					  VT, // float *  	vt
					   n  // int  		ldvt
	);
	#endif
	return retval;
}

int dsvd(double *U, double *S, double *VT, double *Y, const int m, const int n) {
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
	int retval = 0, mn = MIN(m,n);
	#ifdef USE_LAPACK_GESVD
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

int csvd(scomplex_t *U, float *S, scomplex_t *VT, scomplex_t *Y, const int m, const int n) {
	/*
		Single value decomposition (SVD) using Lapack.

		U(m,n)   are the POD modes and must come preallocated.
		S(n)     are the singular values.
		VT(n,n)  are the right singular vectors (transposed).
	*/
	int retval, mn = MIN(m,n);
	#ifdef USE_LAPACK_GESVD
	// Run LAPACKE ZGESVD for the single value decomposition
	float *superb;
	superb = (float*)malloc((mn-1)*sizeof(float));
	retval = LAPACKE_cgesvd(
		LAPACK_ROW_MAJOR, // int  		 matrix_layout
					 'S', // char  		 jobu
					 'S', // char  		 jobvt
					   m, // int  		 m
					   n, // int  		 n
					   Y, // scomplex_t* a
					   n, // int  		 lda
					   S, // scomplex_t* s
					   U, // scomplex_t* u
					  mn, // int  		 ldu
					  VT, // scomplex_t* vt
					   n, // int  		 ldvt
				  superb  // float*      superb
	);
	free(superb);
	#else
	// Run LAPACKE DGESDD for the single value decomposition
	retval = LAPACKE_cgesdd(
		LAPACK_ROW_MAJOR, // int  		  matrix_layout
					 'S', // char  		  jobz
					   m, // int  		  m
					   n, // int  		  n
					   Y, // scomplex_t*  a
					   n, // int  		  lda
					   S, // scomplex_t * s
					   U, // scomplex_t * u
					  mn, // int  		  ldu
					  VT, // scomplex_t * vt
					   n  // int  		  ldvt
	);
	#endif
	return retval;
}

int zsvd(dcomplex_t *U, double *S, dcomplex_t *VT, dcomplex_t *Y, const int m, const int n) {
	/*
		Single value decomposition (SVD) using Lapack.

		U(m,n)   are the POD modes and must come preallocated.
		S(n)     are the singular values.
		VT(n,n)  are the right singular vectors (transposed).
	*/
	int retval, mn = MIN(m,n);
	#ifdef USE_LAPACK_GESVD
	// Run LAPACKE ZGESVD for the single value decomposition
	double *superb;
	superb = (double*)malloc((mn-1)*sizeof(double));
	retval = LAPACKE_zgesvd(
		LAPACK_ROW_MAJOR, // int  		 matrix_layout
					 'S', // char  		 jobu
					 'S', // char  		 jobvt
					   m, // int  		 m
					   n, // int  		 n
					   Y, // dcomplex_t* a
					   n, // int  		 lda
					   S, // dcomplex_t* s
					   U, // dcomplex_t* u
					  mn, // int  		 ldu
					  VT, // dcomplex_t* vt
					   n, // int  		 ldvt
				  superb  // double*     superb
	);
	free(superb);
	#else
	// Run LAPACKE DGESDD for the single value decomposition
	retval = LAPACKE_zgesdd(
		LAPACK_ROW_MAJOR, // int  		  matrix_layout
					 'S', // char  		  jobz
					   m, // int  		  m
					   n, // int  		  n
					   Y, // dcomplex_t*  a
					   n, // int  		  lda
					   S, // dcomplex_t * s
					   U, // dcomplex_t * u
					  mn, // int  		  ldu
					  VT, // dcomplex_t * vt
					   n  // int  	      ldvt
	);
	#endif
	return retval;
}

int sqr(float *Q, float *R, float *A, const int m, const int n) {
	/*
		QR factorization using LAPACK.

		Q(m,n) is the Q matrix and must come preallocated
		R(n,n) is the R matrix and must come preallocated
	*/
	int info = 0, ii, jj;
	float *tau;
	// Allocate
	tau = (float*)malloc(n*n*sizeof(float));
	// Copy A to Q
	memcpy(Q,A,m*n*sizeof(float));
	// Run LAPACK dgerqf - QR factorization on A
	info = LAPACKE_sgeqrf(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
					   m, // int  		m
					   n, // int  		n
					   Q, // float*  	a
					   n, // int  		lda
					 tau  // float * 	tau
	);
	if (!(info==0)) {free(tau); return info;}
	// Copy Ri matrix
	memset(R,0,n*n*sizeof(float));
	for(ii=0;ii<n;++ii)
		for(jj=ii;jj<n;++jj)
			AC_MAT(R,n,ii,jj) = AC_MAT(Q,n,ii,jj);
	// Run LAPACK dorgqr - Generate Q matrix
	info = LAPACKE_sorgqr(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
					   m, // int  		m
					   n, // int  		n
					   n, // int  		k
					   Q, // float*  	a
					   n, // int  		lda
					 tau  // float * 	tau
	);
	if (!(info==0)) {free(tau); return info;}
	free(tau);
	return info;
}

int dqr(double *Q, double *R, double *A, const int m, const int n) {
	/*
		QR factorization using LAPACK.

		Q(m,n) is the Q matrix and must come preallocated
		R(n,n) is the R matrix and must come preallocated
	*/
	int info = 0, ii, jj;
	double *tau;
	// Allocate
	tau = (double*)malloc(n*n*sizeof(double));
	// Copy A to Q
	memcpy(Q,A,m*n*sizeof(double));
	// Run LAPACK dgerqf - QR factorization on A
	info = LAPACKE_dgeqrf(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
					   m, // int  		m
					   n, // int  		n
					   Q, // double*  	a
					   n, // int  		lda
					 tau  // double * 	tau
	);
	if (!(info==0)) {free(tau); return info;}
	// Copy Ri matrix
	memset(R,0,n*n*sizeof(double));
	for(ii=0;ii<n;++ii)
		for(jj=ii;jj<n;++jj)
			AC_MAT(R,n,ii,jj) = AC_MAT(Q,n,ii,jj);
	// Run LAPACK dorgqr - Generate Q matrix
	info = LAPACKE_dorgqr(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
					   m, // int  		m
					   n, // int  		n
					   n, // int  		k
					   Q, // double*  	a
					   n, // int  		lda
					 tau  // double * 	tau
	);
	if (!(info==0)) {free(tau); return info;}
	free(tau);
	return info;
}

int cqr(scomplex_t *Q, scomplex_t *R, scomplex_t *A, const int m, const int n) {
	/*
		QR factorization using LAPACK.

		Q(m,n) is the Q matrix and must come preallocated
		R(n,n) is the R matrix and must come preallocated
	*/
	int info = 0, ii, jj;
	scomplex_t *tau;
	// Allocate
	tau = (scomplex_t*)malloc(n*n*sizeof(scomplex_t));
	// Copy A to Q
	memcpy(Q,A,m*n*sizeof(scomplex_t));
	// Run LAPACK dgerqf - QR factorization on A
	info = LAPACKE_cgeqrf(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
					   m, // int  		m
					   n, // int  		n
					   Q, // scomplex_t* a
					   n, // int  		lda
					 tau  // scomplex_t* tau
	);
	if (!(info==0)) {free(tau); return info;}
	// Copy Ri matrix
	memset(R,0,n*n*sizeof(scomplex_t));
	for(ii=0;ii<n;++ii)
		for(jj=ii;jj<n;++jj)
			AC_MAT(R,n,ii,jj) = AC_MAT(Q,n,ii,jj);
	// Run LAPACK dorgqr - Generate Q matrix
	info = LAPACKE_cungqr(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
					   m, // int  		m
					   n, // int  		n
					   n, // int  		k
					   Q, // scomplex_t* a
					   n, // int  		lda
					 tau  // scomplex_t* tau
	);
	if (!(info==0)) {free(tau); return info;}
	free(tau);
	return info;
}

int zqr(dcomplex_t *Q, dcomplex_t *R, dcomplex_t *A, const int m, const int n) {
	/*
		QR factorization using LAPACK.

		Q(m,n) is the Q matrix and must come preallocated
		R(n,n) is the R matrix and must come preallocated
	*/
	int info = 0, ii, jj;
	dcomplex_t *tau;
	// Allocate
	tau = (dcomplex_t*)malloc(n*n*sizeof(dcomplex_t));
	// Copy A to Q
	memcpy(Q,A,m*n*sizeof(dcomplex_t));
	// Run LAPACK dgerqf - QR factorization on A
	info = LAPACKE_zgeqrf(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
					   m, // int  		m
					   n, // int  		n
					   Q, // dcomplex_t* a
					   n, // int  		lda
					 tau  // dcomplex_t* tau
	);
	if (!(info==0)) {free(tau); return info;}
	// Copy Ri matrix
	memset(R,0,n*n*sizeof(dcomplex_t));
	for(ii=0;ii<n;++ii)
		for(jj=ii;jj<n;++jj)
			AC_MAT(R,n,ii,jj) = AC_MAT(Q,n,ii,jj);
	// Run LAPACK dorgqr - Generate Q matrix
	info = LAPACKE_zungqr(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
					   m, // int  		m
					   n, // int  		n
					   n, // int  		k
					   Q, // dcomplex_t* a
					   n, // int  		lda
					 tau  // dcomplex_t* tau
	);
	if (!(info==0)) {free(tau); return info;}
	free(tau);
	return info;
}

int nextPowerOf2(int n) {
	int p = 1;
	if (n && !(n & (n - 1))) return n;
	while (p < n) p <<= 1;
	return p;
}

int stsqr(float *Qi, float *R, float *Ai, const int m, const int n) {
	/*
		Single value decomposition (SVD) using TSQR algorithm from
		J. Demmel, L. Grigori, M. Hoemmen, and J. Langou, ‘Communication-optimal Parallel
		and Sequential QR and LU Factorizations’, SIAM J. Sci. Comput.,
		vol. 34, no. 1, pp. A206–A239, Jan. 2012,

		doi: 10.1137/080731992.

		Ai(m,n)  data matrix dispersed on each processor.

		Qi(m,n)  Q matrix per processor.
		R(n,n)   R matrix.
	*/
	int info = 0, ii, jj, n2 = n*2, ilevel, blevel, mask;
	int mpi_rank, mpi_size;
	float *Q1i, *Q2i, *Q2l, *QW, *C;
	// Recover rank and size
	MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
	// Memory allocation
	Q1i = (float*)malloc(m*n*sizeof(float));
	Q2i = (float*)malloc(n2*n*sizeof(float));
	QW  = (float*)malloc(n*n*sizeof(float));
	C   = (float*)malloc(n2*n*sizeof(float));
	// Preallocate QW to identity
	memset(QW,0.,n*n*sizeof(float));
	for (ii=0; ii<n; ++ii)
		AC_MAT(QW,n,ii,ii) = 1.;
	// Algorithm 1 from Demmel et al (2012)
	// 1: QR Factorization on Ai to obtain Q1i and Ri
	info = sqr(Q1i,R,Ai,m,n); if (!(info==0)) return info;
	// Reduction, every processor sends R and computes V2i
	int next_power = nextPowerOf2(mpi_size);
	int nlevels    = (int)(log2(next_power));
	int prank;
	Q2l = (float*)malloc(nlevels*n2*n*sizeof(float));
	for (blevel=1,ilevel=0; blevel < next_power; blevel<<=1,++ilevel) {
		// Store R in the upper part of the C matrix
		for (ii=0; ii<n; ++ii)
			for (jj=0; jj<n; ++jj)
				AC_MAT(C,n,ii,jj) = AC_MAT(R,n,ii,jj);
		// Decide who sends and who recieves, use R as buffer
		prank = mpi_rank^blevel;
		if (mpi_rank&blevel) {
			if (prank < mpi_size) MPI_Send(R,n*n,MPI_FLOAT,prank,0,MPI_COMM_WORLD);
		} else {
			// Receive R
			if (prank < mpi_size) {
				MPI_Recv(R,n*n,MPI_FLOAT,prank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				// Store R in the lower part of the C matrix
				for (ii=0; ii<n; ++ii)
					for (jj=0; jj<n; ++jj)
						AC_MAT(C,n,ii+n,jj) = AC_MAT(R,n,ii,jj);
				// 2: QR from the C matrix, reuse C and R
				info = sqr(Q2i,R,C,n2,n); if (!(info==0)) return info;
				// Store Q2i from this level
//				smatmul(Q2i,C,QW,n2,n,n);
				for (ii=0; ii<n2; ++ii)
					for (jj=0; jj<n; ++jj)
						AC_MAT(Q2l,n,ii+ilevel*n2,jj) = AC_MAT(Q2i,n,ii,jj);
			}
		}
	}
	// At this point R is correct on processor 0
	// Broadcast R and its part of the Q matrix
	if (mpi_size > 1) {
		for (blevel = 1 << (nlevels-1),mask=blevel-1,ilevel=nlevels-1; blevel >= 1; blevel>>=1,mask>>=1,--ilevel) {
			if ( ((mpi_rank^0)&mask) == 0 ) {
				// Obtain Q2i for this level - use C as buffer
				for (ii=0; ii<n2; ++ii)
					for (jj=0; jj<n; ++jj)
						AC_MAT(C,n,ii,jj) = AC_MAT(Q2l,n,ii+ilevel*n2,jj);
				// Multiply by QW either set to identity or allocated to a value
				// Store into Q2i
				smatmul(Q2i,C,QW,n2,n,n);
				// Communications scheme
				prank = mpi_rank^blevel;
				if ( ((mpi_rank^0)&blevel)) {
					if (prank < mpi_size) { // Recieve
						MPI_Recv(C,n2*n,MPI_FLOAT,prank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						// Recover R from the upper part of C and QW from the lower part
						for (ii=0; ii<n; ++ii)
							for (jj=0; jj<n; ++jj) {
								AC_MAT(R,n,ii,jj)  = AC_MAT(C,n,ii,jj);
								AC_MAT(QW,n,ii,jj) = AC_MAT(C,n,ii+n,jj);
							}
					}
				} else {
					if (prank < mpi_size) { // Send C
						// Set up C matrix for sending
						// Store R in the upper part and Q2i on the lower part
						// Store Q2i of this rank to QW
						for(ii=0;ii<n;++ii)
							for(jj=ii;jj<n;++jj) {
								AC_MAT(C,n,ii,jj)   = AC_MAT(R,n,ii,jj);
								AC_MAT(C,n,ii+n,jj) = AC_MAT(Q2i,n,ii+n,jj);
								AC_MAT(QW,n,ii,jj)  = AC_MAT(Q2i,n,ii,jj);
							}
						MPI_Send(C,n2*n,MPI_FLOAT,prank,0,MPI_COMM_WORLD);
					}
				}
			}
		}
	}
	// Free memory
	free(Q2i); free(Q2l); free(C);
	// Multiply Q1i and QW to obtain Qi
	smatmul(Qi,Q1i,QW,m,n,n);
	free(Q1i); free(QW);
	return info;
}

int stsqr_svd(float *Ui, float *S, float *VT, float *Ai, const int m, const int n) {
	/*
		Single value decomposition (SVD) using TSQR algorithm from
		J. Demmel, L. Grigori, M. Hoemmen, and J. Langou, ‘Communication-optimal Parallel
		and Sequential QR and LU Factorizations’, SIAM J. Sci. Comput.,
		vol. 34, no. 1, pp. A206–A239, Jan. 2012,

		doi: 10.1137/080731992.

		Ai(m,n)  data matrix dispersed on each processor.

		Ui(m,n)  POD modes dispersed on each processor (must come preallocated).
		S(n)     singular values.
		VT(n,n)  right singular vectors (transposed).
	*/
	int info = 0;
	// Algorithm 1 parallel QR decomposition
	float *Qi, *R;
	R    = (float*)malloc(n*n*sizeof(float));
	Qi   = (float*)malloc(m*n*sizeof(float));
	// Call TSQR routine
	info = stsqr(Qi,R,Ai,m,n);

	// Algorithm 2 from Sayadi and Schmid (2016) - Ui, S and VT
	// At this point we have R and Qi scattered on the processors
	float *Ur;
	Ur = (float*)malloc(n*n*sizeof(float));
	// Call SVD routine
	info = ssvd(Ur,S,VT,R,n,n); if (!(info==0)) return info;
	// Compute Ui = Qi x Ur
	smatmul(Ui,Qi,Ur,m,n,n);
	// Free memory
	free(Ur); free(R); free(Qi);
	return info;
}

int dtsqr(double *Qi, double *R, double *Ai, const int m, const int n) {
	/*
		Single value decomposition (SVD) using TSQR algorithm from
		J. Demmel, L. Grigori, M. Hoemmen, and J. Langou, ‘Communication-optimal Parallel
		and Sequential QR and LU Factorizations’, SIAM J. Sci. Comput.,
		vol. 34, no. 1, pp. A206–A239, Jan. 2012,

		doi: 10.1137/080731992.

		Ai(m,n)  data matrix dispersed on each processor.

		Qi(m,n)  Q matrix per processor.
		R(n,n)   R matrix.
	*/
	int info = 0, ii, jj, n2 = n*2, ilevel, blevel, mask;
	int mpi_rank, mpi_size;
	double *Q1i, *Q2i, *Q2l, *QW, *C;
	// Recover rank and size
	MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
	// Memory allocation
	Q1i = (double*)malloc(m*n*sizeof(double));
	Q2i = (double*)malloc(n2*n*sizeof(double));
	QW  = (double*)malloc(n*n*sizeof(double));
	C   = (double*)malloc(n2*n*sizeof(double));
	// Preallocate QW to identity
	memset(QW,0.,n*n*sizeof(double));
	for (ii=0; ii<n; ++ii)
		AC_MAT(QW,n,ii,ii) = 1.;
	// Algorithm 1 from Demmel et al (2012)
	// 1: QR Factorization on Ai to obtain Q1i and Ri
	info = dqr(Q1i,R,Ai,m,n); if (!(info==0)) return info;
	// Reduction, every processor sends R and computes V2i
	int next_power = nextPowerOf2(mpi_size);
	int nlevels    = (int)(log2(next_power));
	int prank;
	Q2l = (double*)malloc(nlevels*n2*n*sizeof(double));
	for (blevel=1,ilevel=0; blevel < next_power; blevel<<=1,++ilevel) {
		// Store R in the upper part of the C matrix
		for (ii=0; ii<n; ++ii)
			for (jj=0; jj<n; ++jj)
				AC_MAT(C,n,ii,jj) = AC_MAT(R,n,ii,jj);
		// Decide who sends and who recieves, use R as buffer
		prank = mpi_rank^blevel;
		if (mpi_rank&blevel) {
			if (prank < mpi_size) MPI_Send(R,n*n,MPI_DOUBLE,prank,0,MPI_COMM_WORLD);
		} else {
			// Receive R
			if (prank < mpi_size) {
				MPI_Recv(R,n*n,MPI_DOUBLE,prank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				// Store R in the lower part of the C matrix
				for (ii=0; ii<n; ++ii)
					for (jj=0; jj<n; ++jj)
						AC_MAT(C,n,ii+n,jj) = AC_MAT(R,n,ii,jj);
				// 2: QR from the C matrix, reuse C and R
				info = dqr(Q2i,R,C,n2,n); if (!(info==0)) return info;
				// Store Q2i from this level
//				dmatmul(Q2i,C,QW,n2,n,n);
				for (ii=0; ii<n2; ++ii)
					for (jj=0; jj<n; ++jj)
						AC_MAT(Q2l,n,ii+ilevel*n2,jj) = AC_MAT(Q2i,n,ii,jj);
			}
		}
	}
	// At this point R is correct on processor 0
	// Broadcast R and its part of the Q matrix
	if (mpi_size > 1) {
		for (blevel = 1 << (nlevels-1),mask=blevel-1,ilevel=nlevels-1; blevel >= 1; blevel>>=1,mask>>=1,--ilevel) {
			if ( ((mpi_rank^0)&mask) == 0 ) {
				// Obtain Q2i for this level - use C as buffer
				for (ii=0; ii<n2; ++ii)
					for (jj=0; jj<n; ++jj)
						AC_MAT(C,n,ii,jj) = AC_MAT(Q2l,n,ii+ilevel*n2,jj);
				// Multiply by QW either set to identity or allocated to a value
				// Store into Q2i
				dmatmul(Q2i,C,QW,n2,n,n);
				// Communications scheme
				prank = mpi_rank^blevel;
				if ( ((mpi_rank^0)&blevel)) {
					if (prank < mpi_size) { // Recieve
						MPI_Recv(C,n2*n,MPI_DOUBLE,prank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						// Recover R from the upper part of C and QW from the lower part
						for (ii=0; ii<n; ++ii)
							for (jj=0; jj<n; ++jj) {
								AC_MAT(R,n,ii,jj)  = AC_MAT(C,n,ii,jj);
								AC_MAT(QW,n,ii,jj) = AC_MAT(C,n,ii+n,jj);
							}
					}
				} else {
					if (prank < mpi_size) { // Send C
						// Set up C matrix for sending
						// Store R in the upper part and Q2i on the lower part
						// Store Q2i of this rank to QW
						for(ii=0;ii<n;++ii)
							for(jj=ii;jj<n;++jj) {
								AC_MAT(C,n,ii,jj)   = AC_MAT(R,n,ii,jj);
								AC_MAT(C,n,ii+n,jj) = AC_MAT(Q2i,n,ii+n,jj);
								AC_MAT(QW,n,ii,jj)  = AC_MAT(Q2i,n,ii,jj);
							}
						MPI_Send(C,n2*n,MPI_DOUBLE,prank,0,MPI_COMM_WORLD);
					}
				}
			}
		}
	}
	// Free memory
	free(Q2i); free(Q2l); free(C);
	// Multiply Q1i and QW to obtain Qi
	dmatmul(Qi,Q1i,QW,m,n,n);
	free(Q1i); free(QW);
	return info;
}

int dtsqr_svd(double *Ui, double *S, double *VT, double *Ai, const int m, const int n) {
	/*
		Single value decomposition (SVD) using TSQR algorithm from
		J. Demmel, L. Grigori, M. Hoemmen, and J. Langou, ‘Communication-optimal Parallel
		and Sequential QR and LU Factorizations’, SIAM J. Sci. Comput.,
		vol. 34, no. 1, pp. A206–A239, Jan. 2012,

		doi: 10.1137/080731992.

		Ai(m,n)  data matrix dispersed on each processor.

		Ui(m,n)  POD modes dispersed on each processor (must come preallocated).
		S(n)     singular values.
		VT(n,n)  right singular vectors (transposed).
	*/
	int info = 0;
	// Algorithm 1 parallel QR decomposition
	double *Qi, *R;
	R    = (double*)malloc(n*n*sizeof(double));
	Qi   = (double*)malloc(m*n*sizeof(double));
	// Call TSQR routine
	info = dtsqr(Qi,R,Ai,m,n);

	// Algorithm 2 from Sayadi and Schmid (2016) - Ui, S and VT
	// At this point we have R and Qi scattered on the processors
	double *Ur;
	Ur = (double*)malloc(n*n*sizeof(double));
	// Call SVD routine
	info = dsvd(Ur,S,VT,R,n,n); if (!(info==0)) return info;
	// Compute Ui = Qi x Ur
	dmatmul(Ui,Qi,Ur,m,n,n);
	// Free memory
	free(Ur); free(R); free(Qi);
	return info;
}

int ctsqr(scomplex_t *Qi, scomplex_t *R, scomplex_t *Ai, const int m, const int n) {
	/*
		Single value decomposition (SVD) using TSQR algorithm from
		J. Demmel, L. Grigori, M. Hoemmen, and J. Langou, ‘Communication-optimal Parallel
		and Sequential QR and LU Factorizations’, SIAM J. Sci. Comput.,
		vol. 34, no. 1, pp. A206–A239, Jan. 2012,

		doi: 10.1137/080731992.

		Ai(m,n)  data matrix dispersed on each processor.

		Qi(m,n)  Q matrix per processor.
		R(n,n)   R matrix.
	*/
	int info = 0, ii, jj, n2 = n*2, ilevel, blevel, mask;
	int mpi_rank, mpi_size;
	scomplex_t *Q1i, *Q2i, *Q2l, *QW, *C;
	// Recover rank and size
	MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
	// Memory allocation
	Q1i = (scomplex_t*)malloc(m*n*sizeof(scomplex_t));
	Q2i = (scomplex_t*)malloc(n2*n*sizeof(scomplex_t));
	QW  = (scomplex_t*)malloc(n*n*sizeof(scomplex_t));
	C   = (scomplex_t*)malloc(n2*n*sizeof(scomplex_t));
	// Preallocate QW to identity
	memset(QW,0.,n*n*sizeof(scomplex_t));
	for (ii=0; ii<n; ++ii)
		AC_MAT(QW,n,ii,ii) = 1. + 0.*I;
	// Algorithm 1 from Demmel et al (2012)
	// 1: QR Factorization on Ai to obtain Q1i and Ri
	info = cqr(Q1i,R,Ai,m,n); if (!(info==0)) return info;
	// Reduction, every processor sends R and computes V2i
	int next_power = nextPowerOf2(mpi_size);
	int nlevels    = (int)(log2(next_power));
	int prank;
	Q2l = (scomplex_t*)malloc(nlevels*n2*n*sizeof(scomplex_t));
	for (blevel=1,ilevel=0; blevel < next_power; blevel<<=1,++ilevel) {
		// Store R in the upper part of the C matrix
		for (ii=0; ii<n; ++ii)
			for (jj=0; jj<n; ++jj)
				AC_MAT(C,n,ii,jj) = AC_MAT(R,n,ii,jj);
		// Decide who sends and who recieves, use R as buffer
		prank = mpi_rank^blevel;
		if (mpi_rank&blevel) {
			if (prank < mpi_size) MPI_Send(R,n*n,MPI_C_FLOAT_COMPLEX,prank,0,MPI_COMM_WORLD);
		} else {
			// Receive R
			if (prank < mpi_size) {
				MPI_Recv(R,n*n,MPI_C_FLOAT_COMPLEX,prank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				// Store R in the lower part of the C matrix
				for (ii=0; ii<n; ++ii)
					for (jj=0; jj<n; ++jj)
						AC_MAT(C,n,ii+n,jj) = AC_MAT(R,n,ii,jj);
				// 2: QR from the C matrix, reuse C and R
				info = cqr(Q2i,R,C,n2,n); if (!(info==0)) return info;
				// Store Q2i from this level
//				cmatmul(Q2i,C,QW,n2,n,n);
				for (ii=0; ii<n2; ++ii)
					for (jj=0; jj<n; ++jj)
						AC_MAT(Q2l,n,ii+ilevel*n2,jj) = AC_MAT(Q2i,n,ii,jj);
			}
		}
	}
	// At this point R is correct on processor 0
	// Broadcast R and its part of the Q matrix
	if (mpi_size > 1) {
		for (blevel = 1 << (nlevels-1),mask=blevel-1,ilevel=nlevels-1; blevel >= 1; blevel>>=1,mask>>=1,--ilevel) {
			if ( ((mpi_rank^0)&mask) == 0 ) {
				// Obtain Q2i for this level - use C as buffer
				for (ii=0; ii<n2; ++ii)
					for (jj=0; jj<n; ++jj)
						AC_MAT(C,n,ii,jj) = AC_MAT(Q2l,n,ii+ilevel*n2,jj);
				// Multiply by QW either set to identity or allocated to a value
				// Store into Q2i
				cmatmul(Q2i,C,QW,n2,n,n);
				// Communications scheme
				prank = mpi_rank^blevel;
				if ( ((mpi_rank^0)&blevel)) {
					if (prank < mpi_size) { // Recieve
						MPI_Recv(C,n2*n,MPI_C_FLOAT_COMPLEX,prank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						// Recover R from the upper part of C and QW from the lower part
						for (ii=0; ii<n; ++ii)
							for (jj=0; jj<n; ++jj) {
								AC_MAT(R,n,ii,jj)  = AC_MAT(C,n,ii,jj);
								AC_MAT(QW,n,ii,jj) = AC_MAT(C,n,ii+n,jj);
							}
					}
				} else {
					if (prank < mpi_size) { // Send C
						// Set up C matrix for sending
						// Store R in the upper part and Q2i on the lower part
						// Store Q2i of this rank to QW
						for(ii=0;ii<n;++ii)
							for(jj=ii;jj<n;++jj) {
								AC_MAT(C,n,ii,jj)   = AC_MAT(R,n,ii,jj);
								AC_MAT(C,n,ii+n,jj) = AC_MAT(Q2i,n,ii+n,jj);
								AC_MAT(QW,n,ii,jj)  = AC_MAT(Q2i,n,ii,jj);
							}
						MPI_Send(C,n2*n,MPI_C_FLOAT_COMPLEX,prank,0,MPI_COMM_WORLD);
					}
				}
			}
		}
	}
	// Free memory
	free(Q2i); free(Q2l); free(C);
	// Multiply Q1i and QW to obtain Qi
	cmatmul(Qi,Q1i,QW,m,n,n);
	free(Q1i); free(QW);
	return info;
}

int ctsqr_svd(scomplex_t *Ui, float *S, scomplex_t *VT, scomplex_t *Ai, const int m, const int n) {
	/*
		Single value decomposition (SVD) using TSQR algorithm from
		J. Demmel, L. Grigori, M. Hoemmen, and J. Langou, ‘Communication-optimal Parallel
		and Sequential QR and LU Factorizations’, SIAM J. Sci. Comput.,
		vol. 34, no. 1, pp. A206–A239, Jan. 2012,

		doi: 10.1137/080731992.

		Ai(m,n)  data matrix dispersed on each processor.

		Ui(m,n)  POD modes dispersed on each processor (must come preallocated).
		S(n)     singular values.
		VT(n,n)  right singular vectors (transposed).
	*/
	int info = 0;
	// Algorithm 1 parallel QR decomposition
	scomplex_t *Qi, *R;
	R    = (scomplex_t*)malloc(n*n*sizeof(scomplex_t));
	Qi   = (scomplex_t*)malloc(m*n*sizeof(scomplex_t));
	// Call TSQR routine
	info = ctsqr(Qi,R,Ai,m,n);

	// Algorithm 2 from Sayadi and Schmid (2016) - Ui, S and VT
	// At this point we have R and Qi scattered on the processors
	scomplex_t *Ur;
	Ur = (scomplex_t*)malloc(n*n*sizeof(scomplex_t));
	// Call SVD routine
	info = csvd(Ur,S,VT,R,n,n); if (!(info==0)) return info;
	// Compute Ui = Qi x Ur
	cmatmul(Ui,Qi,Ur,m,n,n);
	// Free memory
	free(Ur); free(R); free(Qi);
	return info;
}

int ztsqr(dcomplex_t *Qi, dcomplex_t *R, dcomplex_t *Ai, const int m, const int n) {
	/*
		Single value decomposition (SVD) using TSQR algorithm from
		J. Demmel, L. Grigori, M. Hoemmen, and J. Langou, ‘Communication-optimal Parallel
		and Sequential QR and LU Factorizations’, SIAM J. Sci. Comput.,
		vol. 34, no. 1, pp. A206–A239, Jan. 2012,

		doi: 10.1137/080731992.

		Ai(m,n)  data matrix dispersed on each processor.

		Qi(m,n)  Q matrix per processor.
		R(n,n)   R matrix.
	*/
	int info = 0, ii, jj, n2 = n*2, ilevel, blevel, mask;
	int mpi_rank, mpi_size;
	dcomplex_t *Q1i, *Q2i, *Q2l, *QW, *C;
	// Recover rank and size
	MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
	// Memory allocation
	Q1i = (dcomplex_t*)malloc(m*n*sizeof(dcomplex_t));
	Q2i = (dcomplex_t*)malloc(n2*n*sizeof(dcomplex_t));
	QW  = (dcomplex_t*)malloc(n*n*sizeof(dcomplex_t));
	C   = (dcomplex_t*)malloc(n2*n*sizeof(dcomplex_t));
	// Preallocate QW to identity
	memset(QW,0.,n*n*sizeof(dcomplex_t));
	for (ii=0; ii<n; ++ii)
		AC_MAT(QW,n,ii,ii) = 1. + 0.*I;
	// Algorithm 1 from Demmel et al (2012)
	// 1: QR Factorization on Ai to obtain Q1i and Ri
	info = zqr(Q1i,R,Ai,m,n); if (!(info==0)) return info;
	// Reduction, every processor sends R and computes V2i
	int next_power = nextPowerOf2(mpi_size);
	int nlevels    = (int)(log2(next_power));
	int prank;
	Q2l = (dcomplex_t*)malloc(nlevels*n2*n*sizeof(dcomplex_t));
	for (blevel=1,ilevel=0; blevel < next_power; blevel<<=1,++ilevel) {
		// Store R in the upper part of the C matrix
		for (ii=0; ii<n; ++ii)
			for (jj=0; jj<n; ++jj)
				AC_MAT(C,n,ii,jj) = AC_MAT(R,n,ii,jj);
		// Decide who sends and who recieves, use R as buffer
		prank = mpi_rank^blevel;
		if (mpi_rank&blevel) {
			if (prank < mpi_size) MPI_Send(R,n*n,MPI_C_DOUBLE_COMPLEX,prank,0,MPI_COMM_WORLD);
		} else {
			// Receive R
			if (prank < mpi_size) {
				MPI_Recv(R,n*n,MPI_C_DOUBLE_COMPLEX,prank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				// Store R in the lower part of the C matrix
				for (ii=0; ii<n; ++ii)
					for (jj=0; jj<n; ++jj)
						AC_MAT(C,n,ii+n,jj) = AC_MAT(R,n,ii,jj);
				// 2: QR from the C matrix, reuse C and R
				info = zqr(Q2i,R,C,n2,n); if (!(info==0)) return info;
				// Store Q2i from this level
//				zmatmul(Q2i,C,QW,n2,n,n);
				for (ii=0; ii<n2; ++ii)
					for (jj=0; jj<n; ++jj)
						AC_MAT(Q2l,n,ii+ilevel*n2,jj) = AC_MAT(Q2i,n,ii,jj);
			}
		}
	}
	// At this point R is correct on processor 0
	// Broadcast R and its part of the Q matrix
	if (mpi_size > 1) {
		for (blevel = 1 << (nlevels-1),mask=blevel-1,ilevel=nlevels-1; blevel >= 1; blevel>>=1,mask>>=1,--ilevel) {
			if ( ((mpi_rank^0)&mask) == 0 ) {
				// Obtain Q2i for this level - use C as buffer
				for (ii=0; ii<n2; ++ii)
					for (jj=0; jj<n; ++jj)
						AC_MAT(C,n,ii,jj) = AC_MAT(Q2l,n,ii+ilevel*n2,jj);
				// Multiply by QW either set to identity or allocated to a value
				// Store into Q2i
				zmatmul(Q2i,C,QW,n2,n,n);
				// Communications scheme
				prank = mpi_rank^blevel;
				if ( ((mpi_rank^0)&blevel)) {
					if (prank < mpi_size) { // Recieve
						MPI_Recv(C,n2*n,MPI_C_DOUBLE_COMPLEX,prank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						// Recover R from the upper part of C and QW from the lower part
						for (ii=0; ii<n; ++ii)
							for (jj=0; jj<n; ++jj) {
								AC_MAT(R,n,ii,jj)  = AC_MAT(C,n,ii,jj);
								AC_MAT(QW,n,ii,jj) = AC_MAT(C,n,ii+n,jj);
							}
					}
				} else {
					if (prank < mpi_size) { // Send C
						// Set up C matrix for sending
						// Store R in the upper part and Q2i on the lower part
						// Store Q2i of this rank to QW
						for(ii=0;ii<n;++ii)
							for(jj=ii;jj<n;++jj) {
								AC_MAT(C,n,ii,jj)   = AC_MAT(R,n,ii,jj);
								AC_MAT(C,n,ii+n,jj) = AC_MAT(Q2i,n,ii+n,jj);
								AC_MAT(QW,n,ii,jj)  = AC_MAT(Q2i,n,ii,jj);
							}
						MPI_Send(C,n2*n,MPI_C_DOUBLE_COMPLEX,prank,0,MPI_COMM_WORLD);
					}
				}
			}
		}
	}
	// Free memory
	free(Q2i); free(Q2l); free(C);
	// Multiply Q1i and QW to obtain Qi
	zmatmul(Qi,Q1i,QW,m,n,n);
	free(Q1i); free(QW);
	return info;
}

int ztsqr_svd(dcomplex_t *Ui, double *S, dcomplex_t *VT, dcomplex_t *Ai, const int m, const int n) {
	/*
		Single value decomposition (SVD) using TSQR algorithm from
		J. Demmel, L. Grigori, M. Hoemmen, and J. Langou, ‘Communication-optimal Parallel
		and Sequential QR and LU Factorizations’, SIAM J. Sci. Comput.,
		vol. 34, no. 1, pp. A206–A239, Jan. 2012,

		doi: 10.1137/080731992.

		Ai(m,n)  data matrix dispersed on each processor.

		Ui(m,n)  POD modes dispersed on each processor (must come preallocated).
		S(n)     singular values.
		VT(n,n)  right singular vectors (transposed).
	*/
	int info = 0;
	// Algorithm 1 parallel QR decomposition
	dcomplex_t *Qi, *R;
	R    = (dcomplex_t*)malloc(n*n*sizeof(dcomplex_t));
	Qi   = (dcomplex_t*)malloc(m*n*sizeof(dcomplex_t));
	// Call TSQR routine
	info = ztsqr(Qi,R,Ai,m,n);

	// Algorithm 2 from Sayadi and Schmid (2016) - Ui, S and VT
	// At this point we have R and Qi scattered on the processors
	dcomplex_t *Ur;
	Ur = (dcomplex_t*)malloc(n*n*sizeof(dcomplex_t));
	// Call SVD routine
	info = zsvd(Ur,S,VT,R,n,n); if (!(info==0)) return info;
	// Compute Ui = Qi x Ur
	zmatmul(Ui,Qi,Ur,m,n,n);
	// Free memory
	free(Ur); free(R); free(Qi);
	return info;
}

int srandomized_qr(float *Qi, float *B, float *Ai, const int m, const int n, const int r, const int q, unsigned int seed) {
	/*
		Randomized QR factorization with oversampling and power iterations with the algorithm from 
		Erichson, N. B., Voronin, S., Brunton, S. L., & Kutz, J. N. (2016). 
		Randomized matrix decompositions using R. arXiv preprint arXiv:1608.02148.
		
		Ai(m,n)  data matrix dispersed on each processor.
		Qi(m,r)  
		B (r,n)  
	*/
	int info = 0;
	int ii   = 0;
	// Multiply per a random matrix
	float *omega;
	float *Y;
	omega = (float*)malloc(n*r*sizeof(float));
	Y     = (float*)malloc(m*r*sizeof(float));
	srandom_matrix(omega,n,r,seed);
	smatmul(Y,Ai,omega,m,r,n);
	free(omega); 

	// Transpose A
	float *At;
	At = (float*)malloc(n*m*sizeof(float));
	stranspose(Ai,At,m,n);

	// Do power iterations
	
	float *R, *Q2;
	R  = (float*)malloc(r*r*sizeof(float));
	Q2 = (float*)malloc(n*r*sizeof(float));
	for(ii=0;ii<q;++ii){
		info = stsqr(Qi,R,Y,m,r);
		smatmulp(Q2,At,Qi,n,r,m);
		smatmul(Y,Ai,Q2,m,r,n);
	}
	free(At); free(Q2); 
	
	// Call TSQR routine with the results from the power iterations
	info = stsqr(Qi,R,Y,m,r);
	free(R); free(Y); 

	// Transpose Q
	float *Qt;
	Qt = (float*)malloc(r*m*sizeof(float));
	stranspose(Qi,Qt,m,r);

	// Compute B = Q.T x A
	smatmulp(B,Qt,Ai,r,n,m);
	free(Qt);
	
	return info;
}

int sinit_randomized_qr(float *Qi, float *B, float *Y, float *Ai, const int m, const int n, const int r, const int q, unsigned int seed) {
	/*
		Randomized QR factorization with oversampling and power iterations with the algorithm from 
		Erichson, N. B., Voronin, S., Brunton, S. L., & Kutz, J. N. (2016). 
		Randomized matrix decompositions using R. arXiv preprint arXiv:1608.02148.

		Ai(m,n)  data matrix dispersed on each processor.
		Qi(m,r)  
		B (r,n)  
	*/
	int info = 0;
	int ii   = 0;
	// Multiply per a random matrix
	float *omega;
	omega = (float*)malloc(n*r*sizeof(float));
	srandom_matrix(omega,n,r,seed);
	smatmul(Y,Ai,omega,m,r,n);
	free(omega); 

	// Transpose A
	float *At;
	At = (float*)malloc(n*m*sizeof(float));
	stranspose(Ai,At,m,n);

	// Do power iterations
	
	float *R, *Q2;
	R  = (float*)malloc(r*r*sizeof(float));
	Q2 = (float*)malloc(n*r*sizeof(float));
	for(ii=0;ii<q;++ii){
		info = stsqr(Qi,R,Y,m,r);
		smatmulp(Q2,At,Qi,n,r,m);
		smatmul(Y,Ai,Q2,m,r,n);
	}
	free(At); free(Q2); 
	
	// Call TSQR routine with the results from the power iterations
	info = stsqr(Qi,R,Y,m,r);
	free(R); 

	// Transpose Q
	float *Qt;
	Qt = (float*)malloc(r*m*sizeof(float));
	stranspose(Qi,Qt,m,r);

	// Compute B = Q.T x A
	smatmulp(B,Qt,Ai,r,n,m);
	free(Qt);
	
	return info;
}

int supdate_randomized_qr(float *Q2, float *B2, float *Yn, float *Q1, float *B1, float *Yo, float *Ai, const int m, const int n, const int n1, const int n2, const int r, const int q, unsigned int seed) {
	/*
		Randomized QR factorization with oversampling and power iterations with the algorithm from 
		Erichson, N. B., Voronin, S., Brunton, S. L., & Kutz, J. N. (2016). 
		Randomized matrix decompositions using R. arXiv preprint arXiv:1608.02148.

		Ai(m,n)  data matrix dispersed on each processor.
		Qi(m,r)  
		B (r,n)  
	*/
	int info = 0;
	int ii   = 0;
	// Multiply per a random matrix
	float *omega;
	omega = (float*)malloc(n*r*sizeof(float));
	srandom_matrix(omega,n,r,seed);
	smatmul(Yn,Ai,omega,m,r,n);
	free(omega); 

	// Transpose A
	
	float *At;
	At = (float*)malloc(n*m*sizeof(float));
	stranspose(Ai,At,m,n);

	// Do power iterations
	
	float *R, *O2, *Qpi;
	R   = (float*)malloc(r*r*sizeof(float));
	Qpi = (float*)malloc(m*r*sizeof(float));
	O2  = (float*)malloc(n*r*sizeof(float));
	for(ii=0;ii<q;++ii){
		info = stsqr(Qpi,R,Yn,m,r);
		smatmulp(O2,At,Qpi,n,r,m);
		smatmul(Yn,Ai,O2,m,r,n);
	}
	free(At); free(O2); free(Qpi);
	
	for (int i = 0; i < m; ++i) {
        for (int j = 0; j < r; ++j) {
            AC_MAT(Yn, r, i, j) += AC_MAT(Yo, r, i, j);
        }
    }

	// Call TSQR routine with the results from the power iterations
	info = stsqr(Q2,R,Yn,m,r);
	free(R);

	// Transpose Q2t
	float *Q2t;
	Q2t = (float*)malloc(r*m*sizeof(float));
	stranspose(Q2,Q2t,m,r);

	//Compute B modifier Q2.T*Q1
	float *Q2Q1;
	Q2Q1 = (float*)malloc(r*r*sizeof(float));
	smatmulp(Q2Q1,Q2t,Q1,r,r,m);

	// Modify current B
	float *B2o;
	B2o = (float*)malloc(r*n1*sizeof(float));
	smatmul(B2o,Q2Q1,B1,r,n1,r);
	free(Q2Q1);

	// Compute new chunk of B = Q2.T x A
	float *B2n;
	B2n = (float*)malloc(r*n*sizeof(float));
	smatmulp(B2n,Q2t,Ai,r,n,m);
	free(Q2t);

	// Concatenate B2o and B2n
    for (int i = 0; i < r; ++i) {
        memcpy(B2+i*n2, B2o+i*n1, n1*sizeof(float));
        memcpy(B2+i*n2+n1, B2n+i*n, n*sizeof(float));
    }
	free(B2n); free(B2o);

	return info;
}

int srandomized_svd(float *Ui, float *S, float *VT, float *Ai, const int m, const int n, const int r, const int q, unsigned int seed) {
	/*
		Randomized single value decomposition (SVD) with oversampling and power iterations with the algorithm from 
		Erichson, N. B., Voronin, S., Brunton, S. L., & Kutz, J. N. (2016). 
		Randomized matrix decompositions using R. arXiv preprint arXiv:1608.02148.

		Ai(m,n)  data matrix dispersed on each processor.
		Ui(m,n)  POD modes dispersed on each processor (must come preallocated).
		S(n)     singular values.
		VT(n,n)  right singular vectors (transposed).
	*/
	int info = 0;
	int ii   = 0;
	// Multiply per a random matrix
	float *omega;
	float *Y;
	omega = (float*)malloc(n*r*sizeof(float));
	Y     = (float*)malloc(m*r*sizeof(float));
	srandom_matrix(omega,n,r,seed);
	smatmul(Y,Ai,omega,m,r,n);
	free(omega); 

	// Transpose A
	float *At;
	At = (float*)malloc(n*m*sizeof(float));
	stranspose(Ai,At,m,n);

	// Do power iterations
	
	float *Qi, *R, *Q2;
	R  = (float*)malloc(r*r*sizeof(float));
	Qi = (float*)malloc(m*r*sizeof(float));
	Q2 = (float*)malloc(n*r*sizeof(float));
	for(ii=0;ii<q;++ii){
		info = stsqr(Qi,R,Y,m,r);
		smatmulp(Q2,At,Qi,n,r,m);
		smatmul(Y,Ai,Q2,m,r,n);
	}
	free(At); free(Q2); 
	
	// Call TSQR routine with the results from the power iterations
	info = stsqr(Qi,R,Y,m,r);
	free(R); free(Y); 

	// Transpose Q
	float *Qt;
	Qt = (float*)malloc(r*m*sizeof(float));
	stranspose(Qi,Qt,m,r);

	// Compute B = Q.T x A
	float *B;
	B = (float*)malloc(r*n*sizeof(float));
	smatmulp(B,Qt,Ai,r,n,m);
	free(Qt);

	// Call SVD routine
	float *Ur;
	Ur   = (float*)malloc(r*r*sizeof(float));
	info = ssvd(Ur,S,VT,B,r,n); if (!(info==0)) return info;
	free(B);

	// Compute Ui = Qi x Ur
	smatmul(Ui,Qi,Ur,m,r,r);
	
	free(Ur); free(Qi); 

	return info;
}

int drandomized_qr(double *Qi, double *B, double *Ai, const int m, const int n, const int r, const int q, unsigned int seed) {
	/*
		Randomized QR factorization with oversampling and power iterations with the algorithm from	
		Erichson, N. B., Voronin, S., Brunton, S. L., & Kutz, J. N. (2016). 
		Randomized matrix decompositions using R. arXiv preprint arXiv:1608.02148.

		Ai(m,n)  data matrix dispersed on each processor.
		Qi(m,r)  
		B (r,n)  
	*/
	int info = 0;
	int ii   = 0;
	// Multiply per a random matrix
	double *omega;
	double *Y;
	omega = (double*)malloc(n*r*sizeof(double));
	Y     = (double*)malloc(m*r*sizeof(double));
	drandom_matrix(omega,n,r,seed);
	dmatmul(Y,Ai,omega,m,r,n);
	free(omega); 

	// Transpose A
	double *At;
	At = (double*)malloc(n*m*sizeof(double));
	dtranspose(Ai,At,m,n);

	// Do power iterations
	
	double *R, *Q2;
	R  = (double*)malloc(r*r*sizeof(double));
	Q2 = (double*)malloc(n*r*sizeof(double));
	for(ii=0;ii<q;++ii){
		info = dtsqr(Qi,R,Y,m,r);
		dmatmulp(Q2,At,Qi,n,r,m);
		dmatmul(Y,Ai,Q2,m,r,n);
	}
	free(At); free(Q2); 
	
	// Call TSQR routine with the results from the power iterations
	info = dtsqr(Qi,R,Y,m,r);
	free(R); free(Y); 

	// Transpose Q
	double *Qt;
	Qt = (double*)malloc(r*m*sizeof(double));
	dtranspose(Qi,Qt,m,r);

	// Compute B = Q.T x A
	dmatmulp(B,Qt,Ai,r,n,m);
	free(Qt);

	return info;
}

int dinit_randomized_qr(double *Qi, double *B, double *Y, double *Ai, const int m, const int n, const int r, const int q, unsigned int seed) {
	/*
		Randomized QR factorization with oversampling and power iterations with the algorithm from 
		Erichson, N. B., Voronin, S., Brunton, S. L., & Kutz, J. N. (2016). 
		Randomized matrix decompositions using R. arXiv preprint arXiv:1608.02148.

		Ai(m,n)  data matrix dispersed on each processor.
		Qi(m,r)  
		B (r,n)  
	*/
	int info = 0;
	int ii   = 0;
	// Multiply per a random matrix
	double *omega;
	omega = (double*)malloc(n*r*sizeof(double));
	drandom_matrix(omega,n,r,seed);
	dmatmul(Y,Ai,omega,m,r,n);
	free(omega); 

	// Transpose A
	double *At;
	At = (double*)malloc(n*m*sizeof(double));
	dtranspose(Ai,At,m,n);

	// Do power iterations
	
	double *R, *Q2;
	R  = (double*)malloc(r*r*sizeof(double));
	Q2 = (double*)malloc(n*r*sizeof(double));
	for(ii=0;ii<q;++ii){
		info = dtsqr(Qi,R,Y,m,r);
		dmatmulp(Q2,At,Qi,n,r,m);
		dmatmul(Y,Ai,Q2,m,r,n);
	}
	free(At); free(Q2); 
	
	// Call TSQR routine with the results from the power iterations
	info = dtsqr(Qi,R,Y,m,r);
	free(R);

	// Transpose Q
	double *Qt;
	Qt = (double*)malloc(r*m*sizeof(double));
	dtranspose(Qi,Qt,m,r);

	// Compute B = Q.T x A
	dmatmulp(B,Qt,Ai,r,n,m);
	free(Qt);

	return info;
}

int dupdate_randomized_qr(double *Q2, double *B2, double *Yn, double *Q1, double *B1, double *Yo, double *Ai, const int m, const int n, const int n1, const int n2, const int r, const int q, unsigned int seed) {
	/*
		Randomized QR factorization with oversampling and power iterations with the algorithm from 
		Erichson, N. B., Voronin, S., Brunton, S. L., & Kutz, J. N. (2016). 
		Randomized matrix decompositions using R. arXiv preprint arXiv:1608.02148.

		Ai(m,n)  data matrix dispersed on each processor.
		Qi(m,r)  
		B (r,n)  
	*/
	int info = 0;
	int ii   = 0;
	// Multiply per a random matrix
	double *omega;
	omega = (double*)malloc(n*r*sizeof(double));
	drandom_matrix(omega,n,r,seed);
	dmatmul(Yn,Ai,omega,m,r,n);
	free(omega); 

	// Transpose A
	
	double *At;
	At = (double*)malloc(n*m*sizeof(double));
	dtranspose(Ai,At,m,n);

	// Do power iterations
	
	double *R, *O2, *Qpi;
	R   = (double*)malloc(r*r*sizeof(double));
	Qpi = (double*)malloc(m*r*sizeof(double));
	O2  = (double*)malloc(n*r*sizeof(double));
	for(ii=0;ii<q;++ii){
		info = dtsqr(Qpi,R,Yn,m,r);
		dmatmulp(O2,At,Qpi,n,r,m);
		dmatmul(Yn,Ai,O2,m,r,n);
	}
	free(At); free(O2); free(Qpi);
	
	for (int i = 0; i < m; ++i) {
        for (int j = 0; j < r; ++j) {
            AC_MAT(Yn, r, i, j) += AC_MAT(Yo, r, i, j);
        }
    }

	// Call TSQR routine with the results from the power iterations
	info = dtsqr(Q2,R,Yn,m,r);
	free(R);

	// Transpose Q2t
	double *Q2t;
	Q2t = (double*)malloc(r*m*sizeof(double));
	dtranspose(Q2,Q2t,m,r);

	//Compute B modifier Q2.T*Q1
	double *Q2Q1;
	Q2Q1 = (double*)malloc(r*r*sizeof(double));
	dmatmulp(Q2Q1,Q2t,Q1,r,r,m);

	// Modify current B
	double *B2o;
	B2o = (double*)malloc(r*n1*sizeof(double));
	dmatmul(B2o,Q2Q1,B1,r,n1,r);
	free(Q2Q1);

	// Compute new chunk of B = Q2.T x A
	double *B2n;
	B2n = (double*)malloc(r*n*sizeof(double));
	dmatmulp(B2n,Q2t,Ai,r,n,m);
	free(Q2t);

	// Concatenate B2o and B2n
    for (int i = 0; i < r; ++i) {
        memcpy(B2+i*n2, B2o+i*n1, n1*sizeof(double));
        memcpy(B2+i*n2+n1, B2n+i*n, n*sizeof(double));
    }
	free(B2n); free(B2o);

	return info;
}

int drandomized_svd(double *Ui, double *S, double *VT, double *Ai, const int m, const int n, const int r, const int q, unsigned int seed) {
	/*
		Randomized single value decomposition (SVD) with oversampling and power iterations with the algorithm from 
		Erichson, N. B., Voronin, S., Brunton, S. L., & Kutz, J. N. (2016). 
		Randomized matrix decompositions using R. arXiv preprint arXiv:1608.02148.

		Ai(m,n)  data matrix dispersed on each processor
		Ui(m,n)  POD modes dispersed on each processor (must come preallocated).
		S(n)     singular values.
		VT(n,n)  right singular vectors (transposed).
	*/
	int info = 0;
	int ii   = 0;
	// Multiply per a random matrix
	double *omega;
	double *Y;
	omega = (double*)malloc(n*r*sizeof(double));
	Y     = (double*)malloc(m*r*sizeof(double));
	drandom_matrix(omega,n,r,seed);
	dmatmul(Y,Ai,omega,m,r,n);
	free(omega); 

	// Transpose A
	double *At;
	At = (double*)malloc(n*m*sizeof(double));
	dtranspose(Ai,At,m,n);

	// Do power iterations
	
	double *Qi, *R, *Q2;
	R  = (double*)malloc(r*r*sizeof(double));
	Qi = (double*)malloc(m*r*sizeof(double));
	Q2 = (double*)malloc(n*r*sizeof(double));
	for(ii=0;ii<q;++ii){
		info = dtsqr(Qi,R,Y,m,r);
		dmatmulp(Q2,At,Qi,n,r,m);
		dmatmul(Y,Ai,Q2,m,r,n);
	}
	free(At); free(Q2); 
	
	// Call TSQR routine with the results from the power iterations
	info = dtsqr(Qi,R,Y,m,r);
	free(R); free(Y); 

	// Transpose Q
	double *Qt;
	Qt = (double*)malloc(r*m*sizeof(double));
	dtranspose(Qi,Qt,m,r);

	// Compute B = Q.T x A
	double *B;
	B = (double*)malloc(r*n*sizeof(double));
	dmatmulp(B,Qt,Ai,r,n,m);
	free(Qt);

	// Call SVD routine
	double *Ur;
	Ur   = (double*)malloc(r*r*sizeof(double));
	info = dsvd(Ur,S,VT,B,r,n); if (!(info==0)) return info;
	free(B);

	// Compute Ui = Qi x Ur
	dmatmul(Ui,Qi,Ur,m,r,r);
	
	free(Ur); free(Qi); 

	return info;
}