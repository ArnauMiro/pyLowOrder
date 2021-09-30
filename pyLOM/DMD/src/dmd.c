/*
	DMD - C Functions to compute DMD
*/

#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>

#ifdef USE_MKL
#include "mkl.h"
#include "mkl_lapacke.h"
#else
#include "cblas.h"
#include "lapacke.h"
#endif
#include "matrix.h"
#include "dmd.h"


void compute_eigen(double *delta, double *w, double *veps, double *A, 
	const int m, const int n) {
	/*
		Compute the eigenvalues and eigenvectors of a matrix A using
		LAPACK functions.

		All inputs should come preallocated.

		delta(n)  real eigenvalue part
		w(n)      imaginary eigenvalue part
		veps(n,n) eigenvectors

		A(m,n)   matrix to obtain eigenvalues and eigenvectors from
	*/
	int info;
	double *vl;
	vl = (double*)malloc(1*sizeof(double));
	info = LAPACKE_dgeev(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
		'N',              // char       jobvl
		'V',              // char       jobvr
		n,                // int        n
		A,                // double*    A
		m,                // int        lda
		delta,            // double*    wr
		w,                // double*    wi
		vl,               // double*    vl
		n,                // int        ldvl
		veps,             // double*    vr
		n                 // int        ldvr
	);
	free(vl);
	if( info > 0 ) {
		printf("The algorithm computing eigenvalues failed to converge.\n");
		exit( 1 );
	}
}