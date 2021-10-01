/*
	POD - C Functions to compute POD
*/
#define USE_LAPACK_DGESVD
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>

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
#define AC_X_POD(i,j) X_POD[n*(i)+(j)]
#define AC_X(i,j)     X[n*(i)+(j)]
#define AC_V(i,j)     V[n*(i)+(j)]
#define AC_OUT(i,j)   out[n*(i)+(j)]
#define AC_U(i,j)     U[n*(i)+(j)]
#define AC_UR(i,j)    Ur[N*(i)+(j)]
#define AC_VT(i,j)    VT[n*(i)+(j)]
#define AC_VTR(i,j)   VTr[n*(i)+(j)]


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
			out[ii] += AC_X(ii,jj);
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
			AC_OUT(ii,jj) = AC_X(ii,jj) - X_mean[ii];
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

void compute_svd_truncation(double *Ur, double *Sr, double *VTr,
	double *U, double *S, double *VT, const int m, const int n, const int N) {
	/*
		U(m,n)   are the POD modes and must come preallocated.
		S(n)     are the singular values.
		VT(n,n)  are the right singular vectors (transposed).

		U, S and VT are copied to (they come preallocated):

		Ur(m,N)  are the POD modes and must come preallocated.
		Sr(N)    are the singular values.
		VTr(N,n) are the right singular vectors (transposed).
	*/
	#ifdef USE_OMP
	#pragma omp parallel for shared(U,Ur,S,Sr,VT,VTr) firstprivate(m,n,N)
	#endif
	for (int ii=0;ii<N;++ii) {
		// Copy U into Ur
		for (int jj=0;jj<m;++jj)
			AC_UR(jj,ii) = AC_U(jj,ii);
		// Copy S into Sr
		Sr[ii] = S[ii];
		// Copy VT into VTr
		for (int jj=0;jj<n;++jj)
			AC_VTR(ii,jj) = AC_VT(ii,jj);
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
			PSD[ii] = AC_V(m,ii);
	} else {
		#ifdef USE_OMP
		#pragma omp parallel for shared(PSD,V) firstprivate(m,n)
		#endif
		for (int ii=0;ii<n;++ii)
			PSD[ii] = AC_V(ii,m);
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
		cblas_dscal(n,Sr[ii],&AC_VTR(ii,0),1);
	// Step 2: compute Ur(m,N)*VTr(N,n)
	cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,N,
		1.,Ur,N,VTr,n,0.,X,n);
}


double compute_RMSE(double *X_POD, double *X, const int m, const int n) {
	/*
		Compute and return the Root Meean Square Error and returns it

		X_POD(m, n) is the flow reconstructed with truncated matrices
		X(m,n) is the flow reconstructed
	*/
	double sum1 = 0;
	double norm1 = 0;
	double sum2 = 0;
	double norm2 = 0;
	#ifdef USE_OMP
	#pragma omp parallel for shared(X,X_POD) firstprivate(m,n)
	#endif
	for(int in = 0; in < n; ++in) {
		norm1 = 0;
		norm2 = 0;
		for(int im = 0; im < m; ++im){
			norm1 += POW2(AC_X(in, im) - AC_X_POD(in, im));
			norm2 += POW2(AC_X(in, im));
		}
		sum1 += norm1;
		sum2 += norm2;
	}
	return sqrt(sum1/sum2);
}
