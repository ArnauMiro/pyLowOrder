/*
	Averaging routines
*/

#include "averaging.h"

#define AC_MAT(A,n,i,j) *((A)+(n)*(i)+(j)) 


void stemporal_mean(float *out, float *X, const int m, const int n) {
	/*
		Temporal mean of matrix X(m,n) where m is the spatial coordinates
		and n is the number of snapshots.

		out(m,n) is the output matrix that must have been previously allocated.
	*/
	int ii,jj;
	#ifdef USE_OMP
	#pragma omp parallel for private(ii,jj) shared(out,X) firstprivate(m,n)
	#endif
	for(ii=0; ii<m; ++ii) {
		out[ii] = 0.;
		for(jj=0; jj<n; ++jj)
			out[ii] += AC_MAT(X,n,ii,jj);
		out[ii] /= (float)(n);
	}
}

void dtemporal_mean(double *out, double *X, const int m, const int n) {
	/*
		Temporal mean of matrix X(m,n) where m is the spatial coordinates
		and n is the number of snapshots.

		out(m,n) is the output matrix that must have been previously allocated.
	*/
	int ii,jj;
	#ifdef USE_OMP
	#pragma omp parallel for private(ii,jj) shared(out,X) firstprivate(m,n)
	#endif
	for(ii=0; ii<m; ++ii) {
		out[ii] = 0.;
		for(jj=0; jj<n; ++jj)
			out[ii] += AC_MAT(X,n,ii,jj);
		out[ii] /= (double)(n);
	}
}

void stemporal_variance(float *out, float *X, float *Xmean, const int m, const int n) {
	/*
		Temporal variance of matrix X(m,n) where m is the spatial coordinates
		and n is the number of snapshots.

		out(m) is the output matrix that must have been previously allocated.
	*/
	int ii,jj;
	float diff;
	#ifdef USE_OMP
	#pragma omp parallel for private(ii,jj) shared(out,X) firstprivate(m,n)
	#endif
	for(ii=0; ii<m; ++ii) {
		out[ii] = 0.;
		for(jj=0; jj<n; ++jj){
			diff = AC_MAT(X,n,ii,jj) - Xmean[ii];
			out[ii] += diff*diff;
		}
		out[ii] /= (float)(n);
	}
}

void dtemporal_variance(double *out, double *X, double *Xmean, const int m, const int n) {
	/*
		Temporal variance of matrix X(m,n) where m is the spatial coordinates
		and n is the number of snapshots.

		out(m) is the output matrix that must have been previously allocated.
	*/
	int ii,jj;
	double diff;
	#ifdef USE_OMP
	#pragma omp parallel for private(ii,jj) shared(out,X) firstprivate(m,n)
	#endif
	for(ii=0; ii<m; ++ii) {
		out[ii] = 0.;
		for(jj=0; jj<n; ++jj){
			diff = AC_MAT(X,n,ii,jj) - Xmean[ii];
			out[ii] += diff*diff;
		}
		out[ii] /= (double)(n);
	}
}

void ssubtract_mean(float *out, float *X, float *X_mean, const int m, const int n) {
	/*
		Computes out(m,n) = X(m,n) - X_mean(m) where m is the spatial coordinates
		and n is the number of snapshots.

		out(m,n) is the output matrix that must have been previously allocated.
	*/
	int ii, jj;
	#ifdef USE_OMP
	#pragma omp parallel for collapse(2) private(ii,jj) shared(out,X,X_mean) firstprivate(m,n)
	#endif
	for(ii=0; ii<m; ++ii) {
		for(jj=0; jj<n; ++jj)
			AC_MAT(out,n,ii,jj) = AC_MAT(X,n,ii,jj) - X_mean[ii];
	}
}

void dsubtract_mean(double *out, double *X, double *X_mean, const int m, const int n) {
	/*
		Computes out(m,n) = X(m,n) - X_mean(m) where m is the spatial coordinates
		and n is the number of snapshots.

		out(m,n) is the output matrix that must have been previously allocated.
	*/
	int ii, jj;
	#ifdef USE_OMP
	#pragma omp parallel for collapse(2) private(ii,jj) shared(out,X,X_mean) firstprivate(m,n)
	#endif
	for(ii=0; ii<m; ++ii) {
		for(jj=0; jj<n; ++jj)
			AC_MAT(out,n,ii,jj) = AC_MAT(X,n,ii,jj) - X_mean[ii];
	}
}

void snorm_variance(float *out, float *Xnomean, float *X_var, const int m, const int n) {
	/*
		Computes out(m,n) = Xnomean(m,n)/X_var(m) where m is the spatial coordinates
		and n is the number of snapshots.

		out(m,n) is the output matrix that must have been previously allocated.
	*/
	int ii, jj;
	#ifdef USE_OMP
	#pragma omp parallel for collapse(2) private(ii,jj) shared(out,X,X_mean) firstprivate(m,n)
	#endif
	for(ii=0; ii<m; ++ii) {
		for(jj=0; jj<n; ++jj)
			AC_MAT(out,n,ii,jj) = AC_MAT(Xnomean,n,ii,jj)/X_var[ii];
	}
}

void dnorm_variance(double *out, double *Xnomean, double *X_var, const int m, const int n) {
	/*
		Computes out(m,n) = X(m,n)/X_var(m) where m is the spatial coordinates
		and n is the number of snapshots.

		out(m,n) is the output matrix that must have been previously allocated.
	*/
	int ii, jj;
	#ifdef USE_OMP
	#pragma omp parallel for collapse(2) private(ii,jj) shared(out,X,X_mean) firstprivate(m,n)
	#endif
	for(ii=0; ii<m; ++ii) {
		for(jj=0; jj<n; ++jj)
			AC_MAT(out,n,ii,jj) = AC_MAT(Xnomean,n,ii,jj)/X_var[ii];
	}
}