/*
	Truncation operations
*/

#include <math.h>
#include <string.h>
#include "mpi.h"
#include "vector_matrix.h"

#define AC_MAT(A,n,i,j) *((A)+(n)*(i)+(j))

int compute_truncation_residual(double *S, double res, const int n){
	/*
	Function which computes the accumulative residual of the vector S (of size n) and it
	returns truncation instant according to the desired residual, res, imposed by the user.
	*/
	double accumulative;
	double normS = vector_norm(S,0,n);
	int ii;
	for(ii = 0; ii < n; ++ii){
		accumulative = vector_norm(S,ii,n)/normS;
		if(accumulative < res){
      return ii;
    }
  }
	return n;
}

void compute_truncation(double *Ur, double *Sr, double *VTr, double *U,
	double *S, double *VT, const int m, const int n, const int N){
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
			AC_MAT(Ur,N,ii,jj) = AC_MAT(U,n,ii,jj);
    }
		//Copy S into Sr
		Sr[jj] = S[jj];
		//Copy VT into VTr
		memcpy(VTr+n*jj,VT+n*jj,n*sizeof(double));
  }
}
