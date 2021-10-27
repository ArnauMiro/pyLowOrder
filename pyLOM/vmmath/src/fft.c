/*
	FFT - Fast fourier transform and power density distribution
*/
#include <stdlib.h>
#include <complex.h>
#include <math.h>

#ifdef USE_MKL
#include "mkl_dfti.h"
#else
#include "fftw3.h"
#endif
#include "fft.h"


void fft(double *psd, double *y, const double dt, const int n) {
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
	// Copy y to out and store the frequency on y
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
	// Compute PSD and frequency
	#ifdef USE_OMP
	#pragma omp parallel for shared(out,psd) firstprivate(n)
	#endif
	for (int ii=0; ii<n; ++ii) {
		psd[ii] = (creal(out[ii])*creal(out[ii]) + cimag(out[ii])*cimag(out[ii]))/n; // out*conj(out)/n
		y[ii]   = 1./dt/n*(double)(ii);
	}
	#ifdef USE_MKL
	free(out);
	#else
	fftw_free(out);
	#endif
}