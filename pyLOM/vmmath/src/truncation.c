/*
	Truncation operations
*/

#include <math.h>
#include <string.h>
#include "mpi.h"
#include "vector_matrix.h"
#include "truncation.h"

#define AC_MAT(A,n,i,j) *((A)+(n)*(i)+(j))
#define POW2(x)         ((x)*(x))

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

float senergy(float *A, float *B, const int m, const int n) {
	/*
		Compute reconstruction energy as in:
		Eivazi, H., Le Clainche, S., Hoyas, S., & Vinuesa, R. (2022). 
		Towards extraction of orthogonal and parsimonious non-linear modes from turbulent flows. 
		Expert Systems with Applications, 202, 117038.
		https://doi.org/10.1016
	*/
	int ii, jj;
	float sum1 = 0., norm1 = 0., sum1g = 0.;
	float sum2 = 0., norm2 = 0., sum2g = 0.;
	#ifdef USE_OMP
	#pragma omp parallel for private(ii,jj) shared(A,B) firstprivate(m,n)
	#endif
	for(ii = 0; ii < m; ++ii) {
		norm1 = 0.;
		norm2 = 0.;
		for(jj = 0; jj < n; ++jj){
			norm1 += POW2(AC_MAT(A,n,ii,jj) - AC_MAT(B,n,ii,jj));
			norm2 += POW2(AC_MAT(A,n,ii,jj));
		}
		sum1 += norm1;
		sum2 += norm2;
	}
	// Reduce MPI parallel run
	MPI_Allreduce(&sum1,&sum1g,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&sum2,&sum2g,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
	// Return
	return 1 - sum1g/sum2g;
}

double denergy(double *A, double *B, const int m, const int n) {
	/*
		Compute the Root Meean Square Error (RMSE) between two
		matrices and return it

		A(m,n), B(m,n)
	*/
	int ii, jj;
	double sum1 = 0., norm1 = 0., sum1g = 0.;
	double sum2 = 0., norm2 = 0., sum2g = 0.;
	#ifdef USE_OMP
	#pragma omp parallel for private(ii,jj) shared(A,B) firstprivate(m,n)
	#endif
	for(ii = 0; ii < m; ++ii) {
		norm1 = 0.;
		norm2 = 0.;
		for(jj = 0; jj < n; ++jj){
			norm1 += POW2(AC_MAT(A,n,ii,jj) - AC_MAT(B,n,ii,jj));
			norm2 += POW2(AC_MAT(A,n,ii,jj));
		}
		sum1 += norm1;
		sum2 += norm2;
	}
	// Reduce MPI parallel run
	MPI_Allreduce(&sum1,&sum1g,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&sum2,&sum2g,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	// Return
	return 1 - sum1g/sum2g;
}