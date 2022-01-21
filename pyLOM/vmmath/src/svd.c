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
	int info = 0, ii, jj, mm;
	int mpi_rank, mpi_size;
	double *Qi, *Q1i, *R, *Q2i_p, *Q2i;
	// Recover rank and size
	MPI_Comm_rank(comm,&mpi_rank);
	MPI_Comm_size(comm,&mpi_size);
	// Algorithm 1 from Sayadi and Schmid (2016) - Q and R matrices
	// QR Factorization on Ai to obtain Q1i and Ri
	Q1i = (double*)malloc(m*n*sizeof(double));
	R   = (double*)malloc(n*n*sizeof(double));
	info = qr(Q1i,R,Ai,m,n); if (!(info==0)) return info;
	// MPI_ALLGATHER to obtain Rp
	mm    = mpi_size*n;
	Q2i_p = (double*)malloc(mm*n*sizeof(double));
	MPI_Allgather(R,n*n,MPI_DOUBLE,Q2i_p,n*n,MPI_DOUBLE,comm);
	// QR Factorization Rp to obtain Q2i_p and R (reusing R from above)
	info = qr(Q2i_p,R,Q2i_p,mm,n); if (!(info==0)) return info;
	// Finally compute Qi = Q1i x Q2i
	Q2i = (double*)malloc(n*n*sizeof(double));
	Qi  = (double*)malloc(m*n*sizeof(double));
	for(ii=0;ii<n;++ii)
		for(jj=0;jj<n;++jj)
			AC_MAT(Q2i,n,ii,jj) = AC_MAT(Q2i_p,n,ii+mpi_rank*n,jj);
	matmul(Qi,Q1i,Q2i,m,n,n);
	free(Q2i_p); free(Q1i); free(Q2i);

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


//int nextPowerOf2(int n) {  
//	int p = 1;  
//	if (n && !(n & (n - 1))) return n;  
//	while (p < n) p <<= 1;
//	return p;  
//}
//
//int tsqr_svd2(double *Ui, double *S, double *VT, double *Ai, const int m, const int n, MPI_Comm comm) {
//	/*
//		Single value decomposition (SVD) using TSQR algorithm from
//		J. Demmel, L. Grigori, M. Hoemmen, and J. Langou, ‘Communication-optimal Parallel 
//		and Sequential QR and LU Factorizations’, SIAM J. Sci. Comput., 
//		vol. 34, no. 1, pp. A206–A239, Jan. 2012, 
//
//		doi: 10.1137/080731992.
//
//		Ai(m,n)  data matrix dispersed on each processor.
//
//		Ui(m,n)  POD modes dispersed on each processor (must come preallocated).
//		S(n)     singular values.
//		VT(n,n)  right singular vectors (transposed).
//	*/	
//	int info = 0, ii, jj, ilevel, icom;
//	int mpi_rank, mpi_size, do_qr;
//	// Recover rank and size
//	MPI_Comm_rank(comm,&mpi_rank);
//	MPI_Comm_size(comm,&mpi_size);
//	int next_power = nextPowerOf2(mpi_size);
//	int last_power = next_power >> 1;
//	int strategy[next_power], comm_from[mpi_size], comm_to[mpi_size];
//	// Algorithm 1 from Demmel et al (2012)
//	// 1: QR Factorization on Ai to obtain Q1i and Ri
//	double *Qi, *R, *R2, *Q1i, *Q2i, *C;
//	Qi   = (double*)malloc(m*n*sizeof(double));
//	R    = (double*)malloc(n*n*sizeof(double));
//	R2   = (double*)malloc(n*n*sizeof(double));
//	Q1i  = (double*)malloc(m*n*sizeof(double));
//	Q2i  = (double*)malloc(2*n*n*sizeof(double));
//	C    = (double*)malloc(2*n*n*sizeof(double));
//	info = qr(Qi,R,Ai,m,n); if (!(info==0)) return info;
//	// 2: Loop through the levels
//	for(ilevel=1; ilevel < next_power; ilevel<<=1) {
//		// Steps (not necessarily sequential):
//		// 1. Each processor shares its current R matrix with its neighbor (butterfly all-reduction pattern).
//		// 2. Combine adjacent (n x n) R matrices into (2n x n) matrices.
//		// 3. Find the QR factorization of these new matrices.
//		// 4. Extract R values.
//		// 5. Store local implicit Q matrix.
//		do_qr = 0;
//
//		// Find destination processors for butterfly all reduction
//		icom = 0; 
//		for (ii=0; ii<next_power; ++ii) {
//			if (ii < mpi_size) {
//				comm_from[ii] = -1;
//				comm_to[ii]   = -1;
//			}
//			strategy[ii] = ii^ilevel;
//		}
//
//		// Store previous R matrix - ordering so that the lower rank is above
//		if (mpi_rank < strategy[mpi_rank]) {
//			for(ii=0; ii<n; ++ii)
//				for(jj=0; jj<n; ++jj)
//					AC_MAT(C,n,ii,jj) = AC_MAT(R,n,ii,jj);
//		} else {
//			for(ii=n; ii<2*n; ++ii)
//				for(jj=0; jj<n; ++jj)
//					AC_MAT(C,n,ii,jj) = AC_MAT(R,n,ii-n,jj);
//		}
//
//		if (strategy[mpi_rank] < mpi_size) {
//			// Send & Receive matrices ( use R as a buffer )
//			MPI_Sendrecv(R,n*n,MPI_DOUBLE,strategy[mpi_rank],0,
//				R2,n*n,MPI_DOUBLE,strategy[mpi_rank],0,comm,MPI_STATUS_IGNORE);
//			do_qr = 1;
//		}
//
//		// Find out who did not communicate at this level
//		for (ii=0; ii<mpi_size; ++ii) {
//			if (strategy[ii] > mpi_size-1) {
//				comm_to[icom]   = ii;
//				comm_from[icom] = MIN(strategy[ii] -(next_power-mpi_size),mpi_size-1);
//				icom++;
//			}
//		}
//
//		// Loop the remaining communications at this level
//		for (ii=0; ii<icom; ++ii) {
//			// Recieve from rank == comm_from[ii]
//			if (mpi_rank == comm_to[ii] && mpi_rank != comm_from[ii]) {
//				MPI_Recv(R2,n*n,MPI_DOUBLE,comm_from[ii],0,comm,MPI_STATUS_IGNORE);
//				do_qr = 1;
//				break;
//			}
//			// Send to rank == comm_to[ii]	
//			if (mpi_rank != comm_to[ii] && mpi_rank == comm_from[ii]) {
//				MPI_Send(R,n*n,MPI_DOUBLE,comm_to[ii],0,comm);
//			}
//		}
//
//		if (do_qr) {
//			// Complete C matrix - ordering so that the lower rank is above
//			if (mpi_rank < strategy[mpi_rank]) {
//				for(ii=n; ii<2*n; ++ii)
//					for(jj=0; jj<n; ++jj)
//						AC_MAT(C,n,ii,jj) = AC_MAT(R2,n,ii-n,jj);
//			} else {
//				for(ii=0; ii<n; ++ii)
//					for(jj=0; jj<n; ++jj)
//						AC_MAT(C,n,ii,jj) = AC_MAT(R2,n,ii,jj);
//			}
//
//			// Factor QR of C
//			info = qr(Q2i,R,C,2*n,n); if (!(info==0)) return info;
//		} else {
//			// Ri,k = Ri,k-1
//			// That is, do not change R at this level
//			// Q2i is the identity matrix
//			memset(Q2i,0,2*n*n*sizeof(double));
//			for(ii=0; ii<n; ++ii) {
//				AC_MAT(Q2i,n,ii,ii)   = 1.;
//				AC_MAT(Q2i,n,ii+n,ii) = 1.;
//			}
//		}
//		// Store Qi into Q1i for the next level
//		for(ii=0; ii<m; ++ii)
//			for(jj=0; jj<n; ++jj)
//				AC_MAT(Q1i,n,ii,jj) = AC_MAT(Qi,n,ii,jj);
//		// Accumulate Qi at this tree level
//		if (mpi_rank < strategy[mpi_rank])
//			matmul(Qi,Q1i,&AC_MAT(Q2i,n,0,0),m,n,n);
//		else
//			matmul(Qi,Q1i,&AC_MAT(Q2i,n,n,0),m,n,n);
//	}
//	free(C); free(R2); free(Q1i); free(Q2i);
//
//	// Algorithm 2 from Sayadi and Schmid (2016) - Ui, S and VT
//	// At this point we have R and Qi scattered on the processors
//	double *Ur;
//	Ur = (double*)malloc(n*n*sizeof(double));
//	// Call SVD routine
//	info = svd(Ur,S,VT,R,n,n); if (!(info==0)) return info;
//	// Compute Ui = Qi x Ur
//	matmul(Ui,Qi,Ur,m,n,n);
//	// Free memory
//	free(Ur); free(R); free(Qi);
//	return info;
//}