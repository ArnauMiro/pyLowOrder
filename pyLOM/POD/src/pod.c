/*
	POD - C Functions to compute POD
*/
#define USE_LAPACK_DGESVD
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include "mpi.h"

#ifdef USE_MKL
#include "mkl.h"
#include "mkl_lapacke.h"
#include "mkl_dfti.h"
#else
#include "cblas.h"
#include "lapacke.h"
#include "fftw3.h"
#endif
#include "matrix.h"
#include "pod.h"

#define MIN(a,b)    ((a)<(b)) ? (a) : (b)
#define MAX(a,b)    ((a)>(b)) ? (a) : (b)
#define POW2(x)     ((x)*(x))
// Macros to access flattened matrices
#define AC_MAT(A,n,i,j) *((A) + (n)*(i) + (j))
//#define AC_X_POD(i,j)   AC_MAT(X_POD,n,(i),(j))
//#define AC_X(i,j)       AC_MAT(X,n,(i),(j))
//#define AC_V(i,j)       AC_MAT(V,n,(i),(j))
//#define AC_OUT(i,j)     AC_MAT(out,n,(i),(j))
//#define AC_U(i,j)       AC_MAT(U,n,(i),(j))
//#define AC_VT(i,j)      AC_MAT(VT,n,(i),(j))


void compute_temporal_mean(double *out, double *X, const int m, const int n) {
	/*
		Temporal mean of matrix X(m,n) where m is the spatial coordinates
		and n is the number of snapshots.

		out(m,n) is the output matrix that must have been previously allocated.
	*/
	#ifdef USE_OMP
	#pragma omp parallel for shared(out,X) firstprivate(m,n)
	#endif
	for(int ii=0; ii<m; ++ii) {
		out[ii] = 0.;
		for(int jj=0; jj<n; ++jj)
			out[ii] += AC_MAT(X,n,ii,jj);
		out[ii] /= (double)(n);
	}
}

void subtract_temporal_mean(double *out, double *X, double *X_mean, const int m, const int n) {
	/*
		Computes out(m,n) = X(m,n) - X_mean(m) where m is the spatial coordinates
		and n is the number of snapshots.

		out(m,n) is the output matrix that must have been previously allocated.
	*/
	#ifdef USE_OMP
	#pragma omp parallel for collapse(2) shared(out,X,X_mean) firstprivate(m,n)
	#endif
	for(int ii=0; ii<m; ++ii) {
		for(int jj=0; jj<n; ++jj)
			AC_MAT(out,n,ii,jj) = AC_MAT(X,n,ii,jj) - X_mean[ii];
	}
}

void single_value_decomposition(double *U, double *S, double *VT, double *Y, const int m, const int n) {
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
	int info, mn = MIN(m,n);
	#ifdef USE_LAPACK_DGESVD
	// Run LAPACKE DGESVD for the single value decomposition
	double *superb;
	superb = (double*)malloc((mn-1)*sizeof(double));
	info = LAPACKE_dgesvd(
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
	info = LAPACKE_dgesdd(
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
	if( info > 0 ) {
		printf("The algorithm computing SVD failed to converge.\n");
		exit( 1 );
	}
}

void TSQR_single_value_decomposition(double *Ui, double *S, double *VT, double *Ai, 
	const int m, const int n, MPI_Comm comm) {
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
	int info = 0, mn = MIN(m,n);
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
	// Copy Ri matrix
	memset(R,0,n*n*sizeof(double));
	for(int ii=0;ii<n;++ii)
		for(int jj=ii;jj<n;++jj)
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
	// Copy R matrix - reusing R matrix
	memset(R,0,n*n*sizeof(double));
	for(int ii=0;ii<n;++ii)
		for(int jj=ii;jj<n;++jj)
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
	for(int ii=0;ii<n;++ii)
		for(int jj=0;jj<n;++jj)
			AC_MAT(Rtmp,n,ii,jj) = AC_MAT(Rp,n,ii+mpi_rank*n,jj);
	// Finally compute Qi = Atmp x Rp
	cblas_dgemm(
		CblasRowMajor, // const CBLAS_LAYOUT 	  layout
		 CblasNoTrans, // const CBLAS_TRANSPOSE   TransA
		 CblasNoTrans, // const CBLAS_TRANSPOSE   TransB
		            m, // const CBLAS_INDEX 	  M
		            n, // const CBLAS_INDEX 	  N
		            n, // const CBLAS_INDEX 	  K
		          1.0, // const double 	          alpha
		         Atmp, // const double * 	      A
		            n, // const CBLAS_INDEX 	  lda
	  			 Rtmp, // const double * 	      B
		            n, // const CBLAS_INDEX 	  ldb
		           0., // const double 	          beta
 				   Qi, // double * 	              C
		            n  // const CBLAS_INDEX 	  ldc
	);
	// Free memory
	free(Atmp);
	free(tau);
	free(Rp);
	free(Rtmp);
	// At this point we have R and Qi scattered on the processors
	// Algorithm 2 from Sayadi and Schmid (2016) - Ui, S and VT
	double *Ur;
	Ur = (double*)malloc(n*n*sizeof(double));
	// Call SVD routine
	single_value_decomposition(Ur,S,VT,R,n,n);
	// Compute Ui = Qi x Ur
	cblas_dgemm(
		CblasRowMajor, // const CBLAS_LAYOUT 	  layout
		 CblasNoTrans, // const CBLAS_TRANSPOSE   TransA
		 CblasNoTrans, // const CBLAS_TRANSPOSE   TransB
		            m, // const CBLAS_INDEX 	  M
		            n, // const CBLAS_INDEX 	  N
		            n, // const CBLAS_INDEX 	  K
		          1.0, // const double 	          alpha
		           Qi, // const double * 	      A
		            n, // const CBLAS_INDEX 	  lda
				   Ur, // const double * 	      B
		            n, // const CBLAS_INDEX 	  ldb
		           0., // const double 	          beta
 				   Ui, // double * 	              C
		            n  // const CBLAS_INDEX 	  ldc
	);
	// Free memory
	free(Ur);
	free(R);
	free(Qi);
}

int compute_truncation_residual(double *S, double res, const int n) {
	/*
		Function which computes the accumulative residual of the vector S (of size n) and it
		returns truncation instant according to the desired residual, res, imposed by the user.
	*/
	double accumulative;
	double normS = compute_norm(S,0,n);

	for(int ii=0; ii<n; ++ii){
		accumulative = compute_norm(S,ii,n)/normS;
		if(accumulative < res)
			return ii;
	}
	return n;
}

void compute_truncation(double *Ur, double *Sr, double *VTr, double *U, 
	double *S, double *VT, const int m, const int n, const int N) {
	/*
		U(m,n)   are the POD modes and must come preallocated.
		S(n)     are the singular values.
		VT(n,n)  are the right singular vectors (transposed).

		U, S and VT are copied to (they come preallocated):

		Ur(m,N)  are the POD modes and must come preallocated.
		Sr(N)    are the singular values.
		VTr(N,n) are the right singular vectors (transposed).
	*/
	for (int jj=0;jj<N;++jj) {
		// Copy U into Ur
		for (int ii=0;ii<m;++ii)
			AC_MAT(Ur,N,ii,jj) = AC_MAT(U,n,ii,jj);
		// Copy S into Sr
		Sr[jj] = S[jj];
		// Copy VT into VTr
		for (int ii=0;ii<n;++ii)
			AC_MAT(VTr,n,jj,ii) = AC_MAT(VT,n,jj,ii);
	}
}


void compute_power_spectral_density(double *PSD, double *y, const int n) {
	/*
		Compute FFT and power spectral density (PSD) of an array y of size n.
		Uses MKL or FFTW libraries depending on compilation settings.

		y(n)    is the vector where to compute the PSD (a mode).

		PSD(n)  is the power spectrum of y and must come preallocated.
	*/
	#ifdef USE_MKL
	// Use Intel MKL
	double complex *out;
	out = (double complex*)malloc(n*sizeof(double complex));
	// Copy y to out
	#ifdef USE_OMP
	#pragma omp parallel for shared(out,y) firstprivate(n)
	#endif
	for (int ii=0; ii<n; ++ii)
		out[ii] = y[ii] + 0.*I;
	// Create descriptor
	DFTI_DESCRIPTOR_HANDLE handle;
	DftiCreateDescriptor(&handle,DFTI_DOUBLE,DFTI_COMPLEX,1,n);
	DftiSetValue(handle,DFTI_PLACEMENT,DFTI_INPLACE);
	DftiCommitDescriptor(handle);
	DftiComputeForward(handle,out);
	DftiFreeDescriptor(&handle);
	#else
	// Use FFTW libraries
	fftw_complex *out;
	fftw_plan     p;
	// Allocate output complex array
	out = (fftw_complex*)fftw_malloc(n*sizeof(fftw_complex));
	// Create the FFT plan
	// If your program performs many transforms of the same size and initialization time
	// is not important, use FFTW_MEASURE; otherwise use  FFTW_ESTIMATE.
	p = fftw_plan_dft_r2c_1d(n,y,out,FFTW_ESTIMATE);
	// Execute the plan
	fftw_execute(p);
	// Clean-up
	fftw_destroy_plan(p);
	#endif
	// Compute PSD
	#ifdef USE_OMP
	#pragma omp parallel for shared(out,PSD) firstprivate(n)
	#endif
	for (int ii=0; ii<n; ++ii)
		PSD[ii] = (creal(out[ii])*creal(out[ii]) + cimag(out[ii])*cimag(out[ii]))/n; // out*conj(out)/n
	#ifdef USE_MKL
	free(out);
	#else
	fftw_free(out);
	#endif
}


void compute_power_spectral_density_on_mode(double *PSD, double *V, const int n, const int m, const int transposed) {
	/*
		Compute FFT and power spectral density (PSD) of an array y of size n.
		Uses MKL or FFTW libraries depending on compilation settings.

		V(n,n)      is the modes matrix where to compute the PSD.
		m           is the mode which to compute the PSD.
		trasnposed  is a flag to indicate whether V is transposed or not.

		PSD(n)      is the power spectrum of y and must come preallocated.
	*/
	// Fill the values of y according to the selected mode
	// Reuse PSD as the memory area to work
	if (transposed) {
		#ifdef USE_OMP
		#pragma omp parallel for shared(PSD,V) firstprivate(m,n)
		#endif
		for (int ii=0;ii<n;++ii)
			PSD[ii] = AC_MAT(V,n,m,ii);
	} else {
		#ifdef USE_OMP
		#pragma omp parallel for shared(PSD,V) firstprivate(m,n)
		#endif
		for (int ii=0;ii<n;++ii)
			PSD[ii] = AC_MAT(V,n,ii,m);
	}
	// Compute PSD
	compute_power_spectral_density(PSD,PSD,n);
}


void compute_reconstruct_svd(double *X, double *Ur, double *Sr, double *VTr, const int m, const int n, const int N) {
	/*
		Reconstruct the matrix X given U, S and VT coming from a
		SVD decomposition.

		Ur(m,N)   are the POD modes and must come preallocated.
		Sr(N)     are the singular values.
		VTr(N,n)  are the right singular vectors (transposed).

		X(m,n)    is the reconstructed flow and must come preallocated.

		Inspired in: https://software.intel.com/content/www/us/en/develop/articles/implement-pseudoinverse-of-a-matrix-by-intel-mkl.html
	*/
	// Step 1: compute Sr(N)*VTr(N,n)
	#ifdef USE_OMP
	#pragma omp parallel for shared(Sr,VTr) firstprivate(N,n)
	#endif
	for(int ii=0; ii<N; ++ii)
		cblas_dscal(n,Sr[ii],VTr+n*ii,1);
	// Step 2: compute Ur(m,N)*VTr(N,n)
	cblas_dgemm(
		CblasRowMajor, // const CBLAS_LAYOUT 	  layout
		 CblasNoTrans, // const CBLAS_TRANSPOSE   TransA
		 CblasNoTrans, // const CBLAS_TRANSPOSE   TransB
		            m, // const CBLAS_INDEX 	  M
		            n, // const CBLAS_INDEX 	  N
		            N, // const CBLAS_INDEX 	  K
		          1.0, // const double 	          alpha
		           Ur, // const double * 	      A
		            N, // const CBLAS_INDEX 	  lda
				  VTr, // const double * 	      B
		            n, // const CBLAS_INDEX 	  ldb
		           0., // const double 	          beta
 				    X, // double * 	              C
		            n  // const CBLAS_INDEX 	  ldc
	);
}


double compute_RMSE(double *X_POD, double *X, const int m, const int n, MPI_Comm comm) {
	/*
		Compute and return the Root Meean Square Error and returns it

		X_POD(m,n) is the flow reconstructed with truncated matrices
		X(m,n)     is the flow reconstructed
	*/
	double sum1 = 0., norm1 = 0., sum1g = 0.;
	double sum2 = 0., norm2 = 0., sum2g = 0.;
	#ifdef USE_OMP
	#pragma omp parallel for shared(X,X_POD) firstprivate(m,n)
	#endif
	for(int in = 0; in < n; ++in) {
		norm1 = 0.;
		norm2 = 0.;
		for(int im = 0; im < m; ++im){
			norm1 += POW2(AC_MAT(X,n,in,im) - AC_MAT(X_POD,n,in,im));
			norm2 += POW2(AC_MAT(X,n,in,im));
		}
		sum1 += norm1;
		sum2 += norm2;
	}
	// Reduce MPI parallel run
	MPI_Allreduce(&sum1,&sum1g,1,MPI_DOUBLE,MPI_SUM,comm); 
	MPI_Allreduce(&sum2,&sum2g,1,MPI_DOUBLE,MPI_SUM,comm); 
	return sqrt(sum1g/sum2g);
}
