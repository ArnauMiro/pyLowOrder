/*
	SVD - Singular Value Decomposition of a matrix
*/
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <stdio.h>
#include "mpi.h"
typedef float  _Complex scomplex_t;
typedef double _Complex dcomplex_t;

#ifdef USE_MKL
#define MKL_Complex8  scomplex_t
#define MKL_Complex16 dcomplex_t
#include "mkl.h"
#include "mkl_lapacke.h"
#else
// Due to a bug on OpenBLAS we need to switch the
// SVD computation routine
#define USE_LAPACK_GESVD
#include "cblas.h"
#include "lapacke.h"
#endif

#include "vector_matrix.h"
#include "qr.h"
#include "svd.h"

#define AC_MAT(A,n,i,j) *((A)+(n)*(i)+(j))
#define MIN(a,b)         ((a)<(b)) ? (a) : (b)
#define MAX(a,b)         ((a)>(b)) ? (a) : (b)
#define POW2(x)          ((x)*(x))


int ssvd(float *U, float *S, float *VT, float *Y, const int m, const int n) {
	/*
		Single value decomposition (SVD) using Lapack.

		U(m,n)   are the POD modes and must come preallocated.
		S(n)     are the singular values.
		VT(n,n)  are the right singular vectors (transposed).
	*/
	int retval = 0, mn = MIN(m,n);
	#ifdef USE_LAPACK_GESVD
	// Run LAPACKE SGESVD for the single value decomposition
	float *superb;
	superb = (float*)malloc((mn-1)*sizeof(float));
	retval = LAPACKE_sgesvd(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
					 'S', // char  		jobu
					 'S', // char  		jobvt
					   m, // int  		m
					   n, // int  		n
					   Y, // float*  	a
					   n, // int  		lda
					   S, // float *  	s
					   U, // float *  	u
					  mn, // int  		ldu
					  VT, // float *  	vt
					   n, // int  		ldvt
				  superb  // float *  	superb
	);
	free(superb);
	#else
	// Run LAPACKE SGESDD for the single value decomposition
	retval = LAPACKE_sgesdd(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
					 'S', // char  		jobz
					   m, // int  		m
					   n, // int  		n
					   Y, // float*  	a
					   n, // int  		lda
					   S, // float *  	s
					   U, // float *  	u
					  mn, // int  		ldu
					  VT, // float *  	vt
					   n  // int  		ldvt
	);
	#endif
	return retval;
}

int dsvd(double *U, double *S, double *VT, double *Y, const int m, const int n) {
	/*
		Single value decomposition (SVD) using Lapack.

		U(m,n)   are the POD modes and must come preallocated.
		S(n)     are the singular values.
		VT(n,n)  are the right singular vectors (transposed).

		Lapack dgesvd:
			http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_ga84fdf22a62b12ff364621e4713ce02f2.html
			https://www.netlib.org/lapack/explore-html/d0/dee/lapacke__dgesvd_8c_af31b3cb47f7cc3b9f6541303a2968c9f.html
		Lapack dgesdd (more optimized):
			http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_gad8e0f1c83a78d3d4858eaaa88a1c5ab1.html
			http://www.netlib.org/lapack//explore-html/d3/d23/lapacke__dgesdd_8c_aaf227f107a19ae6021f591c4de5fdbd5.html
		On ROW/COL major:
			https://stackoverflow.com/questions/34698550/understanding-lapack-row-major-and-lapack-col-major-with-lda
			https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/lapacke_dgesvd_row.c.htm
	*/
	int retval = 0, mn = MIN(m,n);
	#ifdef USE_LAPACK_GESVD
	// Run LAPACKE DGESVD for the single value decomposition
	double *superb;
	superb = (double*)malloc((mn-1)*sizeof(double));
	retval = LAPACKE_dgesvd(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
					 'S', // char  		jobu
					 'S', // char  		jobvt
					   m, // int  		m
					   n, // int  		n
					   Y, // double*  	a
					   n, // int  		lda
					   S, // double *  	s
					   U, // double *  	u
					  mn, // int  		ldu
					  VT, // double *  	vt
					   n, // int  		ldvt
				  superb  // double *  	superb
	);
	free(superb);
	#else
	// Run LAPACKE DGESDD for the single value decomposition
	retval = LAPACKE_dgesdd(
		LAPACK_ROW_MAJOR, // int  		matrix_layout
					 'S', // char  		jobz
					   m, // int  		m
					   n, // int  		n
					   Y, // double*  	a
					   n, // int  		lda
					   S, // double *  	s
					   U, // double *  	u
					  mn, // int  		ldu
					  VT, // double *  	vt
					   n  // int  		ldvt
	);
	#endif
	return retval;
}

int csvd(scomplex_t *U, float *S, scomplex_t *VT, scomplex_t *Y, const int m, const int n) {
	/*
		Single value decomposition (SVD) using Lapack.

		U(m,n)   are the POD modes and must come preallocated.
		S(n)     are the singular values.
		VT(n,n)  are the right singular vectors (transposed).
	*/
	int retval, mn = MIN(m,n);
	#ifdef USE_LAPACK_GESVD
	// Run LAPACKE ZGESVD for the single value decomposition
	float *superb;
	superb = (float*)malloc((mn-1)*sizeof(float));
	retval = LAPACKE_cgesvd(
		LAPACK_ROW_MAJOR, // int  		 matrix_layout
					 'S', // char  		 jobu
					 'S', // char  		 jobvt
					   m, // int  		 m
					   n, // int  		 n
					   Y, // scomplex_t* a
					   n, // int  		 lda
					   S, // scomplex_t* s
					   U, // scomplex_t* u
					  mn, // int  		 ldu
					  VT, // scomplex_t* vt
					   n, // int  		 ldvt
				  superb  // float*      superb
	);
	free(superb);
	#else
	// Run LAPACKE DGESDD for the single value decomposition
	retval = LAPACKE_cgesdd(
		LAPACK_ROW_MAJOR, // int  		  matrix_layout
					 'S', // char  		  jobz
					   m, // int  		  m
					   n, // int  		  n
					   Y, // scomplex_t*  a
					   n, // int  		  lda
					   S, // scomplex_t * s
					   U, // scomplex_t * u
					  mn, // int  		  ldu
					  VT, // scomplex_t * vt
					   n  // int  		  ldvt
	);
	#endif
	return retval;
}

int zsvd(dcomplex_t *U, double *S, dcomplex_t *VT, dcomplex_t *Y, const int m, const int n) {
	/*
		Single value decomposition (SVD) using Lapack.

		U(m,n)   are the POD modes and must come preallocated.
		S(n)     are the singular values.
		VT(n,n)  are the right singular vectors (transposed).
	*/
	int retval, mn = MIN(m,n);
	#ifdef USE_LAPACK_GESVD
	// Run LAPACKE ZGESVD for the single value decomposition
	double *superb;
	superb = (double*)malloc((mn-1)*sizeof(double));
	retval = LAPACKE_zgesvd(
		LAPACK_ROW_MAJOR, // int  		 matrix_layout
					 'S', // char  		 jobu
					 'S', // char  		 jobvt
					   m, // int  		 m
					   n, // int  		 n
					   Y, // dcomplex_t* a
					   n, // int  		 lda
					   S, // dcomplex_t* s
					   U, // dcomplex_t* u
					  mn, // int  		 ldu
					  VT, // dcomplex_t* vt
					   n, // int  		 ldvt
				  superb  // double*     superb
	);
	free(superb);
	#else
	// Run LAPACKE DGESDD for the single value decomposition
	retval = LAPACKE_zgesdd(
		LAPACK_ROW_MAJOR, // int  		  matrix_layout
					 'S', // char  		  jobz
					   m, // int  		  m
					   n, // int  		  n
					   Y, // dcomplex_t*  a
					   n, // int  		  lda
					   S, // dcomplex_t * s
					   U, // dcomplex_t * u
					  mn, // int  		  ldu
					  VT, // dcomplex_t * vt
					   n  // int  	      ldvt
	);
	#endif
	return retval;
}

int stsqr_svd(float *Ui, float *S, float *VT, float *Ai, const int m, const int n) {
	/*
		Single value decomposition (SVD) using TSQR algorithm from
		J. Demmel, L. Grigori, M. Hoemmen, and J. Langou, ‘Communication-optimal Parallel
		and Sequential QR and LU Factorizations’, SIAM J. Sci. Comput.,
		vol. 34, no. 1, pp. A206–A239, Jan. 2012,

		doi: 10.1137/080731992.

		Ai(m,n)  data matrix dispersed on each processor.

		Ui(m,n)  POD modes dispersed on each processor (must come preallocated).
		S(n)     singular values.
		VT(n,n)  right singular vectors (transposed).
	*/
	int info = 0;
	// Algorithm 1 parallel QR decomposition
	float *Qi, *R;
	R    = (float*)malloc(n*n*sizeof(float));
	Qi   = (float*)malloc(m*n*sizeof(float));
	// Call TSQR routine
	info = stsqr(Qi,R,Ai,m,n);
	// Algorithm 2 from Sayadi and Schmid (2016) - Ui, S and VT
	// At this point we have R and Qi scattered on the processors
	float *Ur;
	Ur = (float*)malloc(n*n*sizeof(float));
	// Call SVD routine
	info = ssvd(Ur,S,VT,R,n,n); if (!(info==0)) return info;
	free(R);
	// Compute Ui = Qi x Ur
	smatmul(Ui,Qi,Ur,m,n,n);
	// Free memory
	free(Ur); free(Qi);
	return info;
}

int dtsqr_svd(double *Ui, double *S, double *VT, double *Ai, const int m, const int n) {
	/*
		Single value decomposition (SVD) using TSQR algorithm from
		J. Demmel, L. Grigori, M. Hoemmen, and J. Langou, ‘Communication-optimal Parallel
		and Sequential QR and LU Factorizations’, SIAM J. Sci. Comput.,
		vol. 34, no. 1, pp. A206–A239, Jan. 2012,

		doi: 10.1137/080731992.

		Ai(m,n)  data matrix dispersed on each processor.

		Ui(m,n)  POD modes dispersed on each processor (must come preallocated).
		S(n)     singular values.
		VT(n,n)  right singular vectors (transposed).
	*/
	int info = 0;
	// Algorithm 1 parallel QR decomposition
	double *Qi, *R;
	R    = (double*)malloc(n*n*sizeof(double));
	Qi   = (double*)malloc(m*n*sizeof(double));
	// Call TSQR routine
	info = dtsqr(Qi,R,Ai,m,n);
	// Algorithm 2 from Sayadi and Schmid (2016) - Ui, S and VT
	// At this point we have R and Qi scattered on the processors
	double *Ur;
	Ur = (double*)malloc(n*n*sizeof(double));
	// Call SVD routine
	info = dsvd(Ur,S,VT,R,n,n); if (!(info==0)) return info;
	free(R);
	// Compute Ui = Qi x Ur
	dmatmul(Ui,Qi,Ur,m,n,n);
	// Free memory
	free(Ur); free(Qi);
	return info;
}

int ctsqr_svd(scomplex_t *Ui, float *S, scomplex_t *VT, scomplex_t *Ai, const int m, const int n) {
	/*
		Single value decomposition (SVD) using TSQR algorithm from
		J. Demmel, L. Grigori, M. Hoemmen, and J. Langou, ‘Communication-optimal Parallel
		and Sequential QR and LU Factorizations’, SIAM J. Sci. Comput.,
		vol. 34, no. 1, pp. A206–A239, Jan. 2012,

		doi: 10.1137/080731992.

		Ai(m,n)  data matrix dispersed on each processor.

		Ui(m,n)  POD modes dispersed on each processor (must come preallocated).
		S(n)     singular values.
		VT(n,n)  right singular vectors (transposed).
	*/
	int info = 0;
	// Algorithm 1 parallel QR decomposition
	scomplex_t *Qi, *R;
	R    = (scomplex_t*)malloc(n*n*sizeof(scomplex_t));
	Qi   = (scomplex_t*)malloc(m*n*sizeof(scomplex_t));
	// Call TSQR routine
	info = ctsqr(Qi,R,Ai,m,n);
	// Algorithm 2 from Sayadi and Schmid (2016) - Ui, S and VT
	// At this point we have R and Qi scattered on the processors
	scomplex_t *Ur;
	Ur = (scomplex_t*)malloc(n*n*sizeof(scomplex_t));
	// Call SVD routine
	info = csvd(Ur,S,VT,R,n,n); if (!(info==0)) return info;
	free(R);
	// Compute Ui = Qi x Ur
	cmatmul(Ui,Qi,Ur,m,n,n);
	// Free memory
	free(Ur); free(Qi);
	return info;
}

int ztsqr_svd(dcomplex_t *Ui, double *S, dcomplex_t *VT, dcomplex_t *Ai, const int m, const int n) {
	/*
		Single value decomposition (SVD) using TSQR algorithm from
		J. Demmel, L. Grigori, M. Hoemmen, and J. Langou, ‘Communication-optimal Parallel
		and Sequential QR and LU Factorizations’, SIAM J. Sci. Comput.,
		vol. 34, no. 1, pp. A206–A239, Jan. 2012,

		doi: 10.1137/080731992.

		Ai(m,n)  data matrix dispersed on each processor.

		Ui(m,n)  POD modes dispersed on each processor (must come preallocated).
		S(n)     singular values.
		VT(n,n)  right singular vectors (transposed).
	*/
	int info = 0;
	// Algorithm 1 parallel QR decomposition
	dcomplex_t *Qi, *R;
	R    = (dcomplex_t*)malloc(n*n*sizeof(dcomplex_t));
	Qi   = (dcomplex_t*)malloc(m*n*sizeof(dcomplex_t));
	// Call TSQR routine
	info = ztsqr(Qi,R,Ai,m,n);
	// Algorithm 2 from Sayadi and Schmid (2016) - Ui, S and VT
	// At this point we have R and Qi scattered on the processors
	dcomplex_t *Ur;
	Ur = (dcomplex_t*)malloc(n*n*sizeof(dcomplex_t));
	// Call SVD routine
	info = zsvd(Ur,S,VT,R,n,n); if (!(info==0)) return info;
	free(R);
	// Compute Ui = Qi x Ur
	zmatmul(Ui,Qi,Ur,m,n,n);
	// Free memory
	free(Ur); free(Qi);
	return info;
}

int srandomized_svd(float *Ui, float *S, float *VT, float *Ai, const int m, const int n, const int r, const int q, unsigned int seed) {
	/*
		Randomized single value decomposition (SVD) with oversampling and power iterations with the algorithm from 
		Erichson, N. B., Voronin, S., Brunton, S. L., & Kutz, J. N. (2016). 
		Randomized matrix decompositions using R. arXiv preprint arXiv:1608.02148.

		Ai(m,n)  data matrix dispersed on each processor.
		Ui(m,n)  POD modes dispersed on each processor (must come preallocated).
		S(n)     singular values.
		VT(n,n)  right singular vectors (transposed).
	*/
	int info = 0;
	float *Qi, *B;
	Qi = (float*)malloc(m*r*sizeof(float));
	B  = (float*)malloc(r*n*sizeof(float));
	// Algorithm 1 parallel randomized QR decomposition
	info = srandomized_qr(Qi,B,Ai,m,n,r,q,seed);
	// Algorithm 2 from Sayadi and Schmid (2016) - Ui, S and VT
	// At this point we have B and Qi scattered on the processors
	float *Ur;
	Ur   = (float*)malloc(r*r*sizeof(float));
	info = ssvd(Ur,S,VT,B,r,n); if (!(info==0)) return info;
	free(B);
	// Compute Ui = Qi x Ur
	smatmul(Ui,Qi,Ur,m,r,r);
	free(Ur); free(Qi); 

	return info;
}

int drandomized_svd(double *Ui, double *S, double *VT, double *Ai, const int m, const int n, const int r, const int q, unsigned int seed) {
	/*
		Randomized single value decomposition (SVD) with oversampling and power iterations with the algorithm from 
		Erichson, N. B., Voronin, S., Brunton, S. L., & Kutz, J. N. (2016). 
		Randomized matrix decompositions using R. arXiv preprint arXiv:1608.02148.

		Ai(m,n)  data matrix dispersed on each processor
		Ui(m,n)  POD modes dispersed on each processor (must come preallocated).
		S(n)     singular values.
		VT(n,n)  right singular vectors (transposed).
	*/
	int info = 0;
	double *Qi, *B;
	Qi = (double*)malloc(m*r*sizeof(double));
	B  = (double*)malloc(r*n*sizeof(double));
	// Algorithm 1 parallel randomized QR decomposition
	info = drandomized_qr(Qi,B,Ai,m,n,r,q,seed);
	// Algorithm 2 from Sayadi and Schmid (2016) - Ui, S and VT
	// At this point we have B and Qi scattered on the processors
	double *Ur;
	Ur   = (double*)malloc(r*r*sizeof(double));
	info = dsvd(Ur,S,VT,B,r,n); if (!(info==0)) return info;
	free(B);
	// Compute Ui = Qi x Ur
	dmatmul(Ui,Qi,Ur,m,r,r);
	free(Ur); free(Qi); 
	return info;
}