/*
	Truncation operations
*/

#include <math.h>
#include <string.h>
#include "mpi.h"
#include "vector_matrix.h"

#define AC_MAT(A,n,i,j) *((A)+(n)*(i)+(j))

int scompute_truncation_residual(float *S, float res, const int n){
	/*
	Function which computes the accumulative residual of the vector S (of size n) and it
	returns truncation instant according to the desired residual, res, imposed by the user.

	If res < 0  it computes the cummulative energy threshold.
	*/
	float accumulative;
	float normS;
	int ii;

	if (res > 0.) {
		// Residual is positive
		normS = svector_norm(S,0,n);
		for(ii = 0; ii < n; ++ii){
			accumulative = svector_norm(S,ii,n)/normS;
			if(accumulative < res) return ii;
		}
	} else {
		// Residual is positive
		normS = svector_sum(S,0,n);
		accumulative = 0.;
		res = -res;
		for(ii = 0; ii < n; ++ii){
			accumulative += S[ii]/normS;
			if(accumulative > res) return ii;
		}
  	}

	return n;
}

int dcompute_truncation_residual(double *S, double res, const int n){
	/*
	Function which computes the accumulative residual of the vector S (of size n) and it
	returns truncation instant according to the desired residual, res, imposed by the user.
	*/
	double accumulative;
	double normS;
	int ii;

	if (res > 0.) {
		// Residual is positive
		normS = dvector_norm(S,0,n);
		for(ii = 0; ii < n; ++ii){
			accumulative = dvector_norm(S,ii,n)/normS;
			if(accumulative < res) return ii;
		}
	} else {
		// Residual is positive
		normS = dvector_sum(S,0,n);
		accumulative = 0.;
		res = -res;
		for(ii = 0; ii < n; ++ii){
			accumulative += S[ii]/normS;
			if(accumulative > res) return ii+1;
		}
  	}

	return n;
}

void scompute_truncation(float *Ur, float *Sr, float *VTr, float *U,
	float *S, float *VT, const int m, const int n, const int nmod, const int N){
	/*
	U(m,n)   are the POD modes and must come preallocated.
	S(n)     are the singular values.
	VT(n,n)  are the right singular vectors (transposed).

	U, S and VT are copied to (they come preallocated):

	Ur(m,N)  are the POD modes and must come preallocated.
	Sr(N)    are the singular values.
	VTr(N,n) are the right singular vectors (transposed).
	*/
	int jj, ii;
	for(jj = 0; jj < N; ++jj){
		//Copy U into Ur
		for(ii = 0; ii < m; ++ii){
			AC_MAT(Ur,N,ii,jj) = AC_MAT(U,nmod,ii,jj);
    	}
		//Copy S into Sr
		Sr[jj] = S[jj];
		//Copy VT into VTr
		memcpy(VTr+n*jj,VT+n*jj,n*sizeof(float));
  	}
}

void dcompute_truncation(double *Ur, double *Sr, double *VTr, double *U,
	double *S, double *VT, const int m, const int n, const int nmod, const int N){
	/*
	U(m,n)   are the POD modes and must come preallocated.
	S(n)     are the singular values.
	VT(n,n)  are the right singular vectors (transposed).

	U, S and VT are copied to (they come preallocated):

	Ur(m,N)  are the POD modes and must come preallocated.
	Sr(N)    are the singular values.
	VTr(N,n) are the right singular vectors (transposed).
	*/
	int jj, ii;
	for(jj = 0; jj < N; ++jj){
		//Copy U into Ur
		for(ii = 0; ii < m; ++ii){
			AC_MAT(Ur,N,ii,jj) = AC_MAT(U,nmod,ii,jj);
    	}
		//Copy S into Sr
		Sr[jj] = S[jj];
		//Copy VT into VTr
		memcpy(VTr+n*jj,VT+n*jj,n*sizeof(double));
    }
}
