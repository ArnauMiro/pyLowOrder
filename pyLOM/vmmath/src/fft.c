/*
	FFT - Fast fourier transform and power density distribution
*/
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
typedef float  _Complex fcomplex_t;
typedef double _Complex dcomplex_t;

// External FFT dependences
#undef USE_MKL
#ifdef USE_MKL
	#define MKL_Complex8  scomplex_t
	#define MKL_Complex16 dcomplex_t
	#include "mkl_dfti.h"
#elif USE_FFTW3
	#include "fftw3.h"
	#include "nfft3.h"
#else
	#define kiss_fft_scalar double
	#include "kiss_fftr.h"
#endif

#include "fft.h"


void shammwin(float *out, const int N) {
	/*
		Hamming window
	*/
	for(int i=0; i<N; ++i)
		out[i] = 0.54 - 0.46*cos(2.*M_PI*(float)(i)/(float)(N-1));
}

void dhammwin(double *out, const int N) {
	/*
		Hamming window
	*/
	for(int i=0; i<N; ++i)
		out[i] = 0.54 - 0.46*cos(2.*M_PI*(double)(i)/(double)(N-1));
}

void sfft1D(fcomplex_t *out, float *y, const int n) {
	/*
		Compute FFT of an array y of size n.
		Uses MKL or FFTW libraries depending on compilation settings.

		y(n)    is the vector where to compute the FFT.

		out(n)  is the FFT of y and must come preallocated.
	*/
	#ifdef USE_MKL
		// Copy y to out and store the frequency on y
		#ifdef USE_OMP
		#pragma omp parallel for private(ii) shared(out,y) firstprivate(n)
		#endif
		for (int ii=0; ii<n; ++ii)
			out[ii] = y[ii] + 0.*I;
		// Create descriptor
		DFTI_DESCRIPTOR_HANDLE handle;
		DftiCreateDescriptor(&handle,DFTI_SINGLE,DFTI_COMPLEX,1,n);
		DftiSetValue(handle,DFTI_PLACEMENT,DFTI_INPLACE);
		DftiCommitDescriptor(handle);
		DftiComputeForward(handle,out);
		DftiFreeDescriptor(&handle);
	#elif USE_FFTW3
		// Use FFTW libraries
		fftwf_plan     p;
		// Create the FFT plan
		// If your program performs many transforms of the same size and initialization time
		// is not important, use FFTW_MEASURE; otherwise use  FFTW_ESTIMATE.
		p = fftwf_plan_dft_r2c_1d(n,y,out,FFTW_ESTIMATE);
		// Execute the plan
		fftwf_execute(p);
		// Clean-up
		fftwf_destroy_plan(p);
	#else
		// Use KISSFFT libraries
		// KISSFFT does not support single precision, we operate in double precision
		// and convert back to single
		dcomplex_t *aux; aux = (dcomplex_t *)malloc(n*sizeof(dcomplex_t));
		kiss_fftr_cfg cfg = kiss_fftr_alloc(n,0,NULL,NULL);
		kiss_fftr(cfg,(double*)y,(kiss_fft_cpx*)aux);
		kiss_fftr_free(cfg);
		#ifdef USE_OMP
		#pragma omp parallel for private(ii) shared(out,aux) firstprivate(n)
		#endif
		for (int ii=0; ii<n; ++ii)
			out[ii] = (float)creal(aux[ii]) + (float)cimag(aux[ii])*I;
	#endif
}

void dfft1D(double complex *out, double *y, const int n) {
	/*
		Compute FFT of an array y of size n.
		Uses MKL or FFTW libraries depending on compilation settings.

		y(n)    is the vector where to compute the FFT.

		out(n)  is the FFT of y and must come preallocated.
	*/
	#ifdef USE_MKL
		// Copy y to out and store the frequency on y
		#ifdef USE_OMP
		#pragma omp parallel for private(ii) shared(out,y) firstprivate(n)
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
	#elif USE_FFTW3
		// Use FFTW libraries
		fftw_plan     p;
		// Create the FFT plan
		// If your program performs many transforms of the same size and initialization time
		// is not important, use FFTW_MEASURE; otherwise use  FFTW_ESTIMATE.
		p = fftw_plan_dft_r2c_1d(n,y,out,FFTW_ESTIMATE);
		// Execute the plan
		fftw_execute(p);
		// Clean-up
		fftw_destroy_plan(p);
	#else
		// Use KISSFFT libraries
		kiss_fftr_cfg cfg = kiss_fftr_alloc(n,0,NULL,NULL);
		kiss_fftr(cfg,y,(kiss_fft_cpx*)out);
		kiss_fftr_free(cfg);
	#endif
}


void sfft(float *psd, float *y, const float dt, const int n) {
	/*
		Compute FFT and power spectral density (PSD) of an array y of size n.
		Uses MKL or FFTW libraries depending on compilation settings.

		y(n)    is the vector where to compute the PSD (a mode).

		PSD(n)  is the power spectrum of y and must come preallocated.
	*/
	int ii;
	#ifdef USE_FFTW3
		// Use FFTW libraries
		fftwf_complex *out;
		// Allocate output complex array
		out = (fftwf_complex*)fftwf_malloc(n*sizeof(fftwf_complex));
	#else
		scomplex_t *out;
		out = (scomplex_t*)malloc(n*sizeof(scomplex_t));
	#endif
	sfft1D(out, y, n);
	// Compute PSD and frequency
	#ifdef USE_OMP
	#pragma omp parallel for private(ii) shared(out,psd) firstprivate(n)
	#endif
	for (ii=0; ii<n; ++ii) {
		psd[ii] = (creal(out[ii])*creal(out[ii]) + cimag(out[ii])*cimag(out[ii]))/n; // out*conj(out)/n
		y[ii]   = 1./dt/n*(float)(ii);
	}
	#ifdef USE_FFTW3
		fftw_free(out);
	#else
		free(out);
	#endif
}

void dfft(double *psd, double *y, const double dt, const int n) {
	/*
		Compute FFT and power spectral density (PSD) of an array y of size n.
		Uses MKL or FFTW libraries depending on compilation settings.

		y(n)    is the vector where to compute the PSD (a mode).

		PSD(n)  is the power spectrum of y and must come preallocated.
	*/
	int ii;
	#ifdef USE_FFTW3
		// Use FFTW libraries
		fftw_complex *out;
		// Allocate output complex array
		out = (fftw_complex*)fftw_malloc(n*sizeof(fftw_complex));
	#else
		double complex *out;
		out = (double complex*)malloc(n*sizeof(double complex));
	#endif
	dfft1D(out, y, n);
	// Compute PSD and frequency
	#ifdef USE_OMP
	#pragma omp parallel for private(ii) shared(out,psd) firstprivate(n)
	#endif
	for (ii=0; ii<n; ++ii) {
		psd[ii] = (creal(out[ii])*creal(out[ii]) + cimag(out[ii])*cimag(out[ii]))/n; // out*conj(out)/n
		y[ii]   = 1./dt/n*(double)(ii);
	}
	#ifdef USE_FFTW3
		fftw_free(out);
	#else
		free(out);
	#endif
}

void snfft(float *psd, float *y, float* t, const int n) {
	/*
		Compute FFT and power spectral density (PSD) of an array y of size n.
		Uses NFFT libraries.

		t(n)    is the vector containing the time indices.
		y(n)    is the vector where to compute the PSD (a mode).

		PSD(n)  is the power spectrum of y and must come preallocated.
	*/
	#ifdef USE_FFTW3
	int ii;
	float k_left = ((float)(n)-1.)/2.;
	// Create the FFT plan
	nfftf_plan p;
	nfftf_init_1d(&p,n,n);
	// Copy input arrays
	#ifdef USE_OMP
	#pragma omp parallel for private(ii) hared(out,psd) firstprivate(n)
	#endif
	for (ii=0; ii<n; ++ii) {
		p.x[ii] = -0.5 + (float)(ii)/n;
		p.f[ii] = y[ii] + 0.*I;
	}
	// Run NFFT
	if (p.flags & PRE_ONE_PSI) nfftf_precompute_one_psi(&p);
	nfftf_adjoint(&p);
	// Compute PSD and frequency
	#ifdef USE_OMP
	#pragma omp parallel for private(ii) shared(out,psd) firstprivate(n)
	#endif
	for (ii=0; ii<n; ++ii) {
		psd[ii] = (creal(p.f_hat[ii])*creal(p.f_hat[ii]) + cimag(p.f_hat[ii])*cimag(p.f_hat[ii]))/n; // out*conj(out)/n
		y[ii]   = ((float)(ii)-k_left)/t[n-1];
	}
	// Deallocate and finish
	nfftf_finalize(&p);
	#endif
}

void dnfft(double *psd, double *y, double* t, const int n) {
	/*
		Compute FFT and power spectral density (PSD) of an array y of size n.
		Uses NFFT libraries.

		t(n)    is the vector containing the time indices.
		y(n)    is the vector where to compute the PSD (a mode).

		PSD(n)  is the power spectrum of y and must come preallocated.
	*/
	#ifdef USE_FFTW3
	int ii;
	double k_left = ((double)(n)-1.)/2.;
	// Create the FFT plan
	nfft_plan p;
	nfft_init_1d(&p,n,n);
	// Copy input arrays
	#ifdef USE_OMP
	#pragma omp parallel for private(ii) hared(out,psd) firstprivate(n)
	#endif
	for (ii=0; ii<n; ++ii) {
		p.x[ii] = -0.5 + (double)(ii)/n;
		p.f[ii] = y[ii] + 0.*I;
	}
	// Run NFFT
	if (p.flags & PRE_ONE_PSI) nfft_precompute_one_psi(&p);
	nfft_adjoint(&p);
	// Compute PSD and frequency
	#ifdef USE_OMP
	#pragma omp parallel for private(ii) shared(out,psd) firstprivate(n)
	#endif
	for (ii=0; ii<n; ++ii) {
		psd[ii] = (creal(p.f_hat[ii])*creal(p.f_hat[ii]) + cimag(p.f_hat[ii])*cimag(p.f_hat[ii]))/n; // out*conj(out)/n
		y[ii]   = ((double)(ii)-k_left)/t[n-1];
	}
	// Deallocate and finish
	nfft_finalize(&p);
	#endif
}
