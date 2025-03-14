/*
	FFT - Fast fourier transform and power density distribution
*/
#include <complex.h>
typedef float  _Complex scomplex_t;
typedef double _Complex dcomplex_t;
#ifdef USE_MKL
	#define MKL_Complex8  scomplex_t
	#define MKL_Complex16 dcomplex_t
#include "mkl.h"
#endif
// Single precision
void shammwin(float *out, const int N);
void sfft1D(scomplex_t *out, float *y, const int n);
void sfft(float *psd, float *y, const float dt, const int n);
void snfft(float *psd, float *t, float* y, const int n);
// Double precision
void dhammwin(double *out, const int N);
void dfft1D(dcomplex_t *out, double *y, const int n);
void dfft(double *psd, double *y, const double dt, const int n);
void dnfft(double *psd, double *t, double* y, const int n);
// Hints for cython
#ifdef USE_FFTW3
#define _USE_FFTW3 1
#else
#define _USE_FFTW3 0
#endif
