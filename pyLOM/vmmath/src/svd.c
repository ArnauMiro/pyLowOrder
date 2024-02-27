/*
	SVD - Singular Value Decomposition of a matrix
*/
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <stdio.h>
#include "mpi.h"
typedef double _Complex complex_t;

#ifdef USE_MKL
#define MKL_Complex16 complex_t
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

int zsvd(complex_t *U, double *S, complex_t *VT, complex_t *Y, const int m, const int n) {
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
	#ifdef USE_LAPACK_GESVD
	// Run LAPACKE ZGESVD for the single value decomposition
	double *superb;
	superb = (double*)malloc((mn-1)*sizeof(double));
	retval = LAPACKE_zgesvd(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
					 'S', // char  		jobu
					 'S', // char  		jobvt
					   m, // int  		m
					   n, // int  		n
					   Y, // complex_t* a
					   n, // int  		lda
					   S, // complex_t* s
					   U, // complex_t* u
					  mn, // int  		ldu
					  VT, // complex_t* vt
					   n, // int  		ldvt
				  superb  // double* superb
	);
	free(superb);
	#else
	// Run LAPACKE DGESDD for the single value decomposition
	retval = LAPACKE_zgesdd(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
					 'S', // char  		jobz
					   m, // int  		m
					   n, // int  		n
					   Y, // complex_t*  	a
					   n, // int  		lda
					   S, // complex_t *  	s
					   U, // complex_t *  	u
					  mn, // int  		ldu
					  VT, // complex_t *  	vt
					   n  // int  		ldvt
	);
	#endif
	return retval;
}

int qr(double *Q, double *R, double *A, const int m, const int n) {
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

int zqr(complex_t *Q, complex_t *R, complex_t *A, const int m, const int n) {
	/*
		QR factorization using LAPACK.

		Q(m,n) is the Q matrix and must come preallocated
		R(n,n) is the R matrix and must come preallocated
	*/
	int info = 0, ii, jj;
	complex_t *tau;
	// Allocate
	tau = (complex_t*)malloc(n*n*sizeof(complex_t));
	// Copy A to Q
	memcpy(Q,A,m*n*sizeof(complex_t));
	// Run LAPACK dgerqf - QR factorization on A
	info = LAPACKE_zgeqrf(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
					   m, // int  		m
					   n, // int  		n
					   Q, // complex_t* a
					   n, // int  		lda
					 tau  // complex_t* tau
	);
	if (!(info==0)) {free(tau); return info;}
	// Copy Ri matrix
	memset(R,0,n*n*sizeof(complex_t));
	for(ii=0;ii<n;++ii)
		for(jj=ii;jj<n;++jj)
			AC_MAT(R,n,ii,jj) = AC_MAT(Q,n,ii,jj);
	// Run LAPACK dorgqr - Generate Q matrix
	info = LAPACKE_zungqr(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
					   m, // int  		m
					   n, // int  		n
					   n, // int  		k
					   Q, // complex_t* a
					   n, // int  		lda
					 tau  // complex_t* tau
	);
	if (!(info==0)) {free(tau); return info;}
	free(tau);
	return info;
}


int tsqr2(double *Qi, double *R, double *Ai, const int m, const int n, MPI_Comm comm) {
	/*
		Single value decomposition (SVD) using TSQR algorithm from
		T. Sayadi and P. J. Schmid, ‘Parallel data-driven decomposition algorithm
		for large-scale datasets: with application to transitional boundary layers’,
		Theor. Comput. Fluid Dyn., vol. 30, no. 5, pp. 415–428, Oct. 2016

		doi: 10.1007/s00162-016-0385-x

		This is the reduce-broadcast variant of the algorithm from:
		https://cerfacs.fr/wp-content/uploads/2016/03/langou.pdf

		Ai(m,n)  data matrix dispersed on each processor.

		Qi(m,n)  Q matrix per processor.
		R(n,n)   R matrix.
	*/
	int info = 0, ii, jj, mm;
	int mpi_rank, mpi_size;
	double *Q1i, *Q2i_p, *Q2i;
	// Recover rank and size
	MPI_Comm_rank(comm,&mpi_rank);
	MPI_Comm_size(comm,&mpi_size);
	// Algorithm 1 from Sayadi and Schmid (2016) - Q and R matrices
	// QR Factorization on Ai to obtain Q1i and Ri
	Q1i = (double*)malloc(m*n*sizeof(double));
	info = qr(Q1i,R,Ai,m,n); if (!(info==0)) return info;
	// MPI_ALLGATHER to obtain Rp
	mm    = mpi_size*n;
	Q2i_p = (double*)malloc(mm*n*sizeof(double));
	MPI_Allgather(R,n*n,MPI_DOUBLE,Q2i_p,n*n,MPI_DOUBLE,comm);
	// QR Factorization Rp to obtain Q2i_p and R (reusing R from above)
	info = qr(Q2i_p,R,Q2i_p,mm,n); if (!(info==0)) return info;
	// Finally compute Qi = Q1i x Q2i
	Q2i = (double*)malloc(n*n*sizeof(double));
	for(ii=0;ii<n;++ii)
		for(jj=0;jj<n;++jj)
			AC_MAT(Q2i,n,ii,jj) = AC_MAT(Q2i_p,n,ii+mpi_rank*n,jj);
	matmul(Qi,Q1i,Q2i,m,n,n);
	free(Q2i_p); free(Q1i); free(Q2i);
	return info;
}

int tsqr_svd2(double *Ui, double *S, double *VT, double *Ai, const int m, const int n, MPI_Comm comm) {
	/*
		Single value decomposition (SVD) using TSQR algorithm from
		T. Sayadi and P. J. Schmid, ‘Parallel data-driven decomposition algorithm
		for large-scale datasets: with application to transitional boundary layers’,
		Theor. Comput. Fluid Dyn., vol. 30, no. 5, pp. 415–428, Oct. 2016

		doi: 10.1007/s00162-016-0385-x

		This is the reduce-broadcast variant of the algorithm from:
		https://cerfacs.fr/wp-content/uploads/2016/03/langou.pdf

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
	info = tsqr2(Qi,R,Ai,m,n,comm);

	// Algorithm 2 from Sayadi and Schmid (2016) - Ui, S and VT
	// At this point we have R and Qi scattered on the processors
	double *Ur;
	Ur = (double*)malloc(n*n*sizeof(double));
	// Call SVD routine
	info = svd(Ur,S,VT,R,n,n); if (!(info==0)) return info;
	// Compute Ui = Qi x Ur
	matmul(Ui,Qi,Ur,m,n,n);
	// Free memory
	free(Ur); free(R); free(Qi);
	return info;
}


int nextPowerOf2(int n) {
	int p = 1;
	if (n && !(n & (n - 1))) return n;
	while (p < n) p <<= 1;
	return p;
}

int tsqr(double *Qi, double *R, double *Ai, const int m, const int n, MPI_Comm comm) {
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
	MPI_Comm_rank(comm,&mpi_rank);
	MPI_Comm_size(comm,&mpi_size);
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
	info = qr(Q1i,R,Ai,m,n); if (!(info==0)) return info;
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
			if (prank < mpi_size) MPI_Send(R,n*n,MPI_DOUBLE,prank,0,comm);
		} else {
			// Receive R
			if (prank < mpi_size) {
				MPI_Recv(R,n*n,MPI_DOUBLE,prank,0,comm,MPI_STATUS_IGNORE);
				// Store R in the lower part of the C matrix
				for (ii=0; ii<n; ++ii)
					for (jj=0; jj<n; ++jj)
						AC_MAT(C,n,ii+n,jj) = AC_MAT(R,n,ii,jj);
				// 2: QR from the C matrix, reuse C and R
				info = qr(Q2i,R,C,n2,n); if (!(info==0)) return info;
				// Store Q2i from this level
//				matmul(Q2i,C,QW,n2,n,n);
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
				matmul(Q2i,C,QW,n2,n,n);
				// Communications scheme
				prank = mpi_rank^blevel;
				if ( ((mpi_rank^0)&blevel)) {
					if (prank < mpi_size) { // Recieve
						MPI_Recv(C,n2*n,MPI_DOUBLE,prank,0,comm,MPI_STATUS_IGNORE);
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
						MPI_Send(C,n2*n,MPI_DOUBLE,prank,0,comm);
					}
				}
			}
		}
	}
	// Free memory
	free(Q2i); free(Q2l); free(C);
	// Multiply Q1i and QW to obtain Qi
	matmul(Qi,Q1i,QW,m,n,n);
	free(Q1i); free(QW);
	return info;
}

int tsqr_svd(double *Ui, double *S, double *VT, double *Ai, const int m, const int n, MPI_Comm comm) {
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
	info = tsqr(Qi,R,Ai,m,n,comm);

	// Algorithm 2 from Sayadi and Schmid (2016) - Ui, S and VT
	// At this point we have R and Qi scattered on the processors
	double *Ur;
	Ur = (double*)malloc(n*n*sizeof(double));
	// Call SVD routine
	info = svd(Ur,S,VT,R,n,n); if (!(info==0)) return info;
	// Compute Ui = Qi x Ur
	matmul(Ui,Qi,Ur,m,n,n);
	// Free memory
	free(Ur); free(R); free(Qi);
	return info;
}

int ztsqr(complex_t *Qi, complex_t *R, complex_t *Ai, const int m, const int n, MPI_Comm comm) {
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
	complex_t *Q1i, *Q2i, *Q2l, *QW, *C;
	// Recover rank and size
	MPI_Comm_rank(comm,&mpi_rank);
	MPI_Comm_size(comm,&mpi_size);
	// Memory allocation
	Q1i = (complex_t*)malloc(m*n*sizeof(complex_t));
	Q2i = (complex_t*)malloc(n2*n*sizeof(complex_t));
	QW  = (complex_t*)malloc(n*n*sizeof(complex_t));
	C   = (complex_t*)malloc(n2*n*sizeof(complex_t));
	// Preallocate QW to identity
	memset(QW,0.,n*n*sizeof(complex_t));
	for (ii=0; ii<n; ++ii)
		AC_MAT(QW,n,ii,ii) = 1. + 0.*I;
	// Algorithm 1 from Demmel et al (2012)
	// 1: QR Factorization on Ai to obtain Q1i and Ri
	info = zqr(Q1i,R,Ai,m,n); if (!(info==0)) return info;
	// Reduction, every processor sends R and computes V2i
	int next_power = nextPowerOf2(mpi_size);
	int nlevels    = (int)(log2(next_power));
	int prank;
	Q2l = (complex_t*)malloc(nlevels*n2*n*sizeof(complex_t));
	for (blevel=1,ilevel=0; blevel < next_power; blevel<<=1,++ilevel) {
		// Store R in the upper part of the C matrix
		for (ii=0; ii<n; ++ii)
			for (jj=0; jj<n; ++jj)
				AC_MAT(C,n,ii,jj) = AC_MAT(R,n,ii,jj);
		// Decide who sends and who recieves, use R as buffer
		prank = mpi_rank^blevel;
		if (mpi_rank&blevel) {
			if (prank < mpi_size) MPI_Send(R,n*n,MPI_C_DOUBLE_COMPLEX,prank,0,comm);
		} else {
			// Receive R
			if (prank < mpi_size) {
				MPI_Recv(R,n*n,MPI_C_DOUBLE_COMPLEX,prank,0,comm,MPI_STATUS_IGNORE);
				// Store R in the lower part of the C matrix
				for (ii=0; ii<n; ++ii)
					for (jj=0; jj<n; ++jj)
						AC_MAT(C,n,ii+n,jj) = AC_MAT(R,n,ii,jj);
				// 2: QR from the C matrix, reuse C and R
				info = zqr(Q2i,R,C,n2,n); if (!(info==0)) return info;
				// Store Q2i from this level
//				matmul(Q2i,C,QW,n2,n,n);
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
						MPI_Recv(C,n2*n,MPI_C_DOUBLE_COMPLEX,prank,0,comm,MPI_STATUS_IGNORE);
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
						MPI_Send(C,n2*n,MPI_C_DOUBLE_COMPLEX,prank,0,comm);
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

int ztsqr_svd(complex_t *Ui, double *S, complex_t *VT, complex_t *Ai, const int m, const int n, MPI_Comm comm) {
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
	complex_t *Qi, *R;
	R    = (complex_t*)malloc(n*n*sizeof(complex_t));
	Qi   = (complex_t*)malloc(m*n*sizeof(complex_t));
	// Call TSQR routine
	info = ztsqr(Qi,R,Ai,m,n,comm);

	// Algorithm 2 from Sayadi and Schmid (2016) - Ui, S and VT
	// At this point we have R and Qi scattered on the processors
	complex_t *Ur;
	Ur = (complex_t*)malloc(n*n*sizeof(complex_t));
	// Call SVD routine
	info = zsvd(Ur,S,VT,R,n,n); if (!(info==0)) return info;
	// Compute Ui = Qi x Ur
	zmatmul(Ui,Qi,Ur,m,n,n);
	// Free memory
	free(Ur); free(R); free(Qi);
	return info;
}
