/*
	POD - C Functions to compute POD
*/

#include <stdlib.h>

#ifdef USE_MKL
#include "mkl_lapacke.h"
#else
#include <complex.h>
#include "lapacke.h"
#include "fftw3.h"
#endif
#include "pod.h"

// Macros to access flattened matrices
#define MIN(a,b)    ((a)<(b)) ? (a) : (b)
#define MAX(a,b)    ((a)>(b)) ? (a) : (b)
#define AC_X(i,j)   X[n*(i)+(j)]
#define AC_V(i,j)   V[n*(i)+(j)]
#define AC_OUT(i,j) out[n*(i)+(j)]


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
	for(int ii=0; ii<m; ++ii){
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
	*/
	#ifdef USE_LAPACK_DGESVD
	// Run LAPACKE DGESVD for the single value decomposition
	double *superb;
	superb = (double*)malloc((int)(MIN(m,n)-1)*sizeof(double));
	LAPACKE_dgesvd(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
					 'S', // char  		jobu
					 'S', // char  		jobvt
					   m, // int  		m
					   n, // int  		n
					   Y, // double*  	a
					   n, // int  		lda
					   S, // double *  	s
					   U, // double *  	u
					   n, // int  		ldu
					  VT, // double *  	vt
					   n, // int  		ldvt
				  superb  // double *  	superb
	);
	free(superb);
	#else
	// Run LAPACKE DGESDD for the single value decomposition
	LAPACKE_dgesdd(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
					 'S', // char  		jobz
					   m, // int  		m
					   n, // int  		n
					   Y, // double*  	a
					   n, // int  		lda
					   S, // double *  	s
					   U, // double *  	u
					   n, // int  		ldu
					  VT, // double *  	vt
					   n  // int  		ldvt
	);
	#endif
}

int compute_truncation_residual(double *S, double res, int n) {
	/*
		TODO...

		returns truncation instant
	*/
}


void compute_svd_truncation(double *U, double *S, double *VT, double *Y, const int m, const int n, const int N) {
	/*
		U(m,n)   are the POD modes and must come preallocated.
		S(n)     are the singular values.
		VT(n,n)  are the right singular vectors (transposed).

		U, S and VT are reallocated to:
		
		U(m,N)   are the POD modes and must come preallocated.
		S(N)     are the singular values.
		VT(N,N)  are the right singular vectors (transposed).
	*/
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
	// TODO
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
	// Compute PSD
	#ifdef USE_OMP
	#pragma omp parallel for shared(out,PSD) firstprivate(n)
	#endif
	for (int ii=0; ii<n; ++ii)
		PSD[ii] = (creal(out[ii])*creal(out[ii]) + cimag(out[ii])*cimag(out[ii]))/n; // out*conj(out)/n

	// Clean-up
	fftw_destroy_plan(p);
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