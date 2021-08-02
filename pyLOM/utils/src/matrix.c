/*
	Matrix math operations
*/

#include <math.h>

#define BLK_LIM     5000
#define AC_MAT(i,j) A[n*(i)+(j)]
#define POW2(x)     ((x)*(x))

void transpose_naive(double *A, const int m, const int n) {
	/*
		Naive approximation to matrix transpose.
		Overwrites A matrix
	*/
	double swp;
	for (int ii=0; ii<m; ++ii) {
		for (int jj=0; jj<ii+1; ++jj) {
			swp = AC_MAT(ii,jj);
			AC_MAT(ii,jj) = AC_MAT(jj,ii);
			AC_MAT(jj,ii) = swp;
		}
	}
}

//template<class T>
//inline matrixMN<T> matrixMN<T>::t_f() { // Fast transpose (M = N)
//	matrixMN<T> out(this->m);
//	// Loop by blocks
//	for (int ib=0; ib<this->m/this->bsz; ++ib) {
//		for(int jb=0; jb<this->n/this->bsz; ++jb) {
//			// Loop matrix
//			for(int i=ib*this->bsz; i<(ib+1)*this->bsz; ++i) {
//				for(int j=jb*this->bsz; j<(jb+1)*this->bsz; ++j) {
//					out.ij(j,i, this->ij(i,j));
//				}
//			}// Loop matrix
//		}
//	}
//
//	return out;
//}

void transpose(double *A, const int m, const int n, const int bsz) {
	/*
		Matrix transpose
	*/
//	if (m==n && BSZ>0 && m>BLK_LIM)
//		transpose_fast(A,m,n);
//	else
		transpose_naive(A,m,n);
}


double compute_norm(double *v, int start, int n){
	/*
		Compute the norm of the n-dim vector v from the position start
	*/
	double norm = 0;
	#ifdef USE_OMP
	#pragma omp parallel for reduction(+:norm) shared(v) firstprivate(start,n)
	#endif
	for(int ii = start; ii < n; ++ii)
		norm += POW2(v[ii]);
	return sqrt(norm);
}

void reorder_matrix(double *A, int M, int N, int n){
	/*Function which reorders the matrix A, of size Mxn, in order to delete the values
	that do not belong to the first N columns.
	Memory has to be reallocated after using the function*/
	int ii = 0;
	for(int im = 0; im < M; ++im){
		for(int in = 0; in < N; ++in){
			A[ii] = A[n*im + in];
			++ii;
		}
	}
}
