/*
	Linear operator
*/
#include <math.h>
#include <stdbool.h>
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

void sconcatenate(float *Y, float *Z, float **X, int *n_list, const int n_matrix, const int m, const int n_total, int remove_mean) {

    int ii, jj, n_jj; 

    float *X_mean;
    float *X_meanless;
    int n_sum = 0;
    X_mean = (float*)malloc(m*sizeof(float));

    for (jj=0; jj<n_matrix; ++jj){
        n_jj = n_list[jj];
	    X_meanless = (float*)malloc(m*n_jj*sizeof(float));

        if (remove_mean){
            stemporal_mean(X_mean, X[jj], m, n_jj);
            ssubtract_mean(X_meanless, X[jj], X_mean, m, n_jj);

            for (ii=0; ii<m; ++ii){
                memcpy(&Y[ii*n_total+n_sum], &X_meanless[ii*n_jj], (n_jj-1)*sizeof(float));
                memcpy(&Z[ii*n_total+n_sum], &X_meanless[ii*n_jj+1], (n_jj-1)*sizeof(float));
            }
        }
        else {
            memcpy(&Y[ii*n_total+n_sum], &X[jj][ii*n_jj], (n_jj-1)*sizeof(float));
            memcpy(&Z[ii*n_total+n_sum], &X[jj][ii*n_jj+1], (n_jj-1)*sizeof(float));
        }
        n_sum += n_jj-1;
        free(X_meanless);
    }
    free(X_mean);
}

void dconcatenate(double *Y, double *Z, double **X, int *n_list, const int n_matrix, const int m, const int n_total, int remove_mean) {

    int ii, jj, n_jj; 

    double *X_mean;
    double *X_meanless;
    int n_sum = 0;
    X_mean = (double*)malloc(m*sizeof(double));

    for (jj=0; jj<n_matrix; ++jj){
        n_jj = n_list[jj];
	    X_meanless = (double*)malloc(m*n_jj*sizeof(double));

        if (remove_mean){
            dtemporal_mean(X_mean, X[jj], m, n_jj);
            dsubtract_mean(X_meanless, X[jj], X_mean, m, n_jj);

            for (ii=0; ii<m; ++ii){
                memcpy(&Y[ii*n_total+n_sum], &X_meanless[ii*n_jj], (n_jj-1)*sizeof(double));
                memcpy(&Z[ii*n_total+n_sum], &X_meanless[ii*n_jj+1], (n_jj-1)*sizeof(double));
            }
        }
        else {
            memcpy(&Y[ii*n_total+n_sum], &X[jj][ii*n_jj], (n_jj-1)*sizeof(double));
            memcpy(&Z[ii*n_total+n_sum], &X[jj][ii*n_jj+1], (n_jj-1)*sizeof(double));
        }
        n_sum += n_jj-1;
        free(X_meanless);
    }
    free(X_mean);
}

void sseparate(float *Y, float *Z, float *X, const int m, const int n, int remove_mean) {

    int ii;

    float *X_mean;
    float *X_meanless;
    X_mean       = (float*)malloc(m*sizeof(float));
	X_meanless   = (float*)malloc(m*n*sizeof(float));

    if (remove_mean){
        stemporal_mean(X_mean, X, m, n);
        ssubtract_mean(X_meanless, X, X_mean, m, n);
        
        for (ii=0; ii<m; ++ii){
            memcpy(&Y[ii*(n-1)], &X_meanless[ii*n], (n-1)*sizeof(float));
            memcpy(&Z[ii*(n-1)], &X_meanless[ii*n+1], (n-1)*sizeof(float));
        }
    }
    else {
        memcpy(&Y[ii*(n-1)], &X[ii*n], (n-1)*sizeof(float));
        memcpy(&Z[ii*(n-1)], &X[ii*n+1], (n-1)*sizeof(float));
    }
}

void dseparate(double *Y, double *Z, double *X, const int m, const int n, int remove_mean) {

    int m_aux, ii;

    double *X_mean;
    double *X_meanless;
    X_mean       = (double*)malloc(m*sizeof(double));
	X_meanless   = (double*)malloc(m*n*sizeof(double));

    if (remove_mean){
        dtemporal_mean(X_mean, X, m, n);
        dsubtract_mean(X_meanless, X, X_mean, m, n);
        
        for (ii=0; ii<m; ++ii){
            memcpy(&Y[ii*(n-1)], &X_meanless[ii*n], (n-1)*sizeof(double));
            memcpy(&Z[ii*(n-1)], &X_meanless[ii*n+1], (n-1)*sizeof(double));
        }
    }
    else {
        memcpy(&Y[ii*(n-1)], &X[ii*n], (n-1)*sizeof(double));
        memcpy(&Z[ii*(n-1)], &X[ii*+1], (n-1)*sizeof(double));
    }
}

void cresolvent(scomplex_t *U, float *S, scomplex_t *V, float *A, scomplex_t w, const int n) {

    int ii, jj;

    scomplex_t *H_inv;
    H_inv       = (scomplex_t*)malloc(n*n*sizeof(scomplex_t));

    // Compte els signes
    for (ii=0; ii<n; ++ii){
        for (jj=0; jj<n; ++jj){
            if (ii == jj) H_inv[ii*n + jj] = w - A[ii*n + jj];
            else H_inv[ii*n + jj] = -A[ii*n + jj];
        }
    }

    scomplex_t *UT_flip;
    scomplex_t *V_flip;
    scomplex_t *U_flip;
    float *S_inv;
    UT_flip       = (scomplex_t*)malloc(n*n*sizeof(scomplex_t));
    V_flip        = (scomplex_t*)malloc(n*n*sizeof(scomplex_t));
    U_flip        = (scomplex_t*)malloc(n*n*sizeof(scomplex_t));
    S_inv         = (float*)malloc(n*sizeof(float));

    csvd(V_flip, S_inv, UT_flip, H_inv, n, n);
    free(H_inv);

    for (ii=0; ii<n; ++ii){
        S[ii] = 1 / S_inv[n-1-ii];
    }
    free(S_inv);

    cdagger(UT_flip, U_flip, n, n);
	free(UT_flip);

    cflip_columns(U_flip, U, n, n);
	cflip_columns(V_flip, V, n, n);
	free(U_flip);
	free(V_flip);
}

void zresolvent(dcomplex_t *U, double *S, dcomplex_t *V, double *A, dcomplex_t w, const int n) {

    int ii, jj;

    dcomplex_t *H_inv;
    H_inv       = (dcomplex_t*)malloc(n*n*sizeof(dcomplex_t));

    // Compte els signes
    for (ii=0; ii<n; ++ii){
        for (jj=0; jj<n; ++jj){
            if (ii == jj) H_inv[ii*n + jj] = w - A[ii*n + jj];
            else H_inv[ii*n + jj] = -A[ii*n + jj];
        }
    }

    dcomplex_t *UT_flip;
    dcomplex_t *V_flip;
    dcomplex_t *U_flip;
    double *S_inv;
    UT_flip       = (dcomplex_t*)malloc(n*n*sizeof(dcomplex_t));
    V_flip        = (dcomplex_t*)malloc(n*n*sizeof(dcomplex_t));
    U_flip        = (dcomplex_t*)malloc(n*n*sizeof(dcomplex_t));
    S_inv         = (double*)malloc(n*sizeof(double));

    zsvd(V_flip, S_inv, UT_flip, H_inv, n, n);
    free(H_inv);

    for (ii=0; ii<n; ++ii){
        S[ii] = 1 / S_inv[n-1-ii];
    }
    free(S_inv);

    zdagger(UT_flip, U_flip, n, n);
	free(UT_flip);

    zflip_columns(U_flip, U, n, n);
	zflip_columns(V_flip, V, n, n);
	free(U_flip);
	free(V_flip);
}