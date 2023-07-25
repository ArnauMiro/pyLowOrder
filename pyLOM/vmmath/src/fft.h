/*
	FFT - Fast fourier transform and power density distribution
*/
#include <complex.h>
typedef double _Complex complex_t;
#ifdef USE_MKL
#include "mkl.h"
#define MKL_Complex16 complex_t
#endif
void fft1D(complex_t *out, double *y, const int n);
void fft(double *psd, double *y, const double dt, const int n);
void nfft(double *psd, double *t, double* y, const int n);