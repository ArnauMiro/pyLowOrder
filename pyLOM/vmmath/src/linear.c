/*
	Linear operator
*/
#include <math.h>
#include <complex.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"
typedef float  _Complex scomplex_t;
typedef double _Complex dcomplex_t;

#ifdef USE_MKL
#define MKL_Complex8  scomplex_t
#define MKL_Complex16 dcomplex_t
#include "mkl.h"
#include "mkl_lapacke.h"
#else
#include "cblas.h"
#include "lapacke.h"
#endif

#include "averaging.h"
#include "vector_matrix.h"
#include "truncation.h"
#include "svd.h"
#include "linear.h"

#define AC_MAT(A,n,i,j) *((A)+(n)*(i)+(j))

int slinear_operator(float *U, float *S, float *VT, float *Atilde, float *Y, float *Z, const float r, const int my, const int mz, const int nn) {
    int icol, irow, retval;

    // Compute SVD
	float *U_aux;
	float *S_all;
	float *VT_all;
	U_aux  = (float*)malloc(my*nn*sizeof(float));
	S_all  = (float*)malloc(nn*sizeof(float));
	VT_all = (float*)malloc(nn*nn*sizeof(float)); 
	retval = stsqr_svd(U_aux, S_all, VT_all, Y, my, nn);
    free(Y);

    float *U_all;
    U_all  = (float*)malloc(mz*nn*sizeof(float));
    sremove_rows(U_aux, U_all, mz, nn);
    free(U_aux);

    // Truncate
    int nr;
    if (r > 1) nr = r;
    else scompute_truncation_residual(S_all, r, nn);
    scompute_truncation(U, S, VT, U_all, S_all, VT_all, mz, nn, nn, nr);

    free(U_all);
    free(S_all);
    free(VT_all);

    // Project Jacobian of the snapshots into the POD basis
    float *aux1, *aux2, *aux3, *Urt;
    aux1   = (float*)malloc(nr*nn*sizeof(float));
	aux2   = (float*)malloc(nr*nn*sizeof(float));
	aux3   = (float*)malloc(nr*sizeof(float));
	Urt    = (float*)malloc(nr*mz*sizeof(float));
    stranspose(U, Urt, mz, nr);
    smatmulp(aux1, Urt, Z, nr, nn, mz);
    free(Z);

    for (icol=0; icol<nn; ++icol){
        for (irow=0; irow<nr; ++irow){
            aux2[icol*nr + irow] = VT[irow*nn + icol]/S[irow];
        }
    }
    smatmul(Atilde, aux1, aux2, nr, nr, nn);
    free(aux1);
    free(aux3);
    free(Urt);

    return retval;
}

int dlinear_operator(double *U, double *S, double *VT, double *Atilde, double *Y, double *Z, const double r, const int my, const int mz, const int nn) {
    int icol, irow, retval;

    // Compute SVD
	double *U_aux;
	double *S_all;
	double *VT_all;
	U_aux  = (double*)malloc(my*nn*sizeof(double));
	S_all  = (double*)malloc(nn*sizeof(double));
	VT_all = (double*)malloc(nn*nn*sizeof(double)); 
	retval = dtsqr_svd(U_aux, S_all, VT_all, Y, my, nn);
    free(Y);

    double *U_all;
    U_all  = (double*)malloc(mz*nn*sizeof(double));
    dremove_rows(U_aux, U_all, mz, nn);
    free(U_aux);

    // Truncate
    int nr;
    if (r > 1) nr = r;
    else dcompute_truncation_residual(S_all, r, nn);
    dcompute_truncation(U, S, VT, U_all, S_all, VT_all, mz, nn, nn, nr);

    free(U_all);
    free(S_all);
    free(VT_all);

    // Project Jacobian of the snapshots into the POD basis
    double *aux1, *aux2, *aux3, *Urt;
    aux1   = (double*)malloc(nr*nn*sizeof(double));
	aux2   = (double*)malloc(nr*nn*sizeof(double));
	aux3   = (double*)malloc(nr*sizeof(double));
	Urt    = (double*)malloc(nr*mz*sizeof(double));
    dtranspose(U, Urt, mz, nr);
    dmatmulp(aux1, Urt, Z, nr, nn, mz);
    free(Z);

    for (icol=0; icol<nn; ++icol){
        for (irow=0; irow<nr; ++irow){
            aux2[icol*nr + irow] = VT[irow*nn + icol]/S[irow];
        }
    }
    dmatmul(Atilde, aux1, aux2, nr, nr, nn);
    free(aux1);
    free(aux3);
    free(Urt);

    return retval;
}

void sflip_columns(float *A, float *B, int m, int n) {

    int ii, jj;
    // #ifdef USE_OMP
	// #pragma omp parallel for collapse(2) private(ii,jj) shared(A,B) firstprivate(m,n)
	// #endif
    for (ii=0; ii<m; ++ii){
        for (jj=0; jj<n; ++jj){
            AC_MAT(B,n,ii,jj) = AC_MAT(A,n,ii,n-jj-1);
        }
    }
}

void dflip_columns(double *A, double *B, int m, int n) {

    int ii, jj;
    // #ifdef USE_OMP
	// #pragma omp parallel for collapse(2) private(ii,jj) shared(A,B) firstprivate(m,n)
	// #endif
    for (ii=0; ii<m; ++ii){
        for (jj=0; jj<n; ++jj){
            AC_MAT(B,n,ii,jj) = AC_MAT(A,n,ii,n-jj-1);
        }
    }
}

void cflip_columns(scomplex_t *A, scomplex_t *B, int m, int n) {

    int ii, jj;
    // #ifdef USE_OMP
	// #pragma omp parallel for collapse(2) private(ii,jj) shared(A,B) firstprivate(m,n)
	// #endif
    for (ii=0; ii<m; ++ii){
        for (jj=0; jj<n; ++jj){
            AC_MAT(B,n,ii,jj) = AC_MAT(A,n,ii,n-jj-1);
        }
    }
}

void zflip_columns(dcomplex_t *A, dcomplex_t *B, int m, int n) {

    int ii, jj;
    // #ifdef USE_OMP
	// #pragma omp parallel for collapse(2) private(ii,jj) shared(A,B) firstprivate(m,n)
	// #endif
    for (ii=0; ii<m; ++ii){
        for (jj=0; jj<n; ++jj){
            AC_MAT(B,n,ii,jj) = AC_MAT(A,n,ii,n-jj-1);
        }
    }
}