/*
	Averaging routines
*/

#include "averaging.h"

#define AC_MAT(A,n,i,j) *((A)+(n)*(i)+(j)) 


void temporal_mean(double *out, double *X, const int m, const int n) {
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

void subtract_mean(double *out, double *X, double *X_mean, const int m, const int n) {
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
