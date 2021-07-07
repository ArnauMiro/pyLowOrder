/*
	POD - C Functions to compute POD
*/

// Macros to access flattened matrices
#define AC_X(i,j)   X[n*ii+jj]
#define AC_OUT(i,j) out[n*ii+jj]

void compute_temporal_mean(double *out, double *X, const int m, const int n) {
	/*
		Temporal mean of matrix X(m,n) where m is the spatial coordinates
		and n is the number of snapshots.

		out is a matrix of out(m) that must have been previously allocated.
	*/
}

void subtract_temporal_mean(double *out, double *X, double *X_mean, const int m, const int n) {
	/*
		Computes out(m,n) = X(m,n) - X_mean(m) where m is the spatial coordinates
		and n is the number of snapshots.
		
		out is a matrix of out(m,n) that must have been previously allocated.
	*/
}

void single_value_decomposition(double *PSI, double *S, double *V, double *Y, const int m, const int n) {
	/*
		Single value decomposition (SVD) using Lapack.

		PSI(m,n) are the POD modes and must come preallocated.
		S(n)     are the singular values.
		V(n,n)   are the right singular vectors.
	*/
}