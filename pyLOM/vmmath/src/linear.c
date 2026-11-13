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

        if (remove_mean){
            X_meanless = (float*)malloc(m*n_jj*sizeof(float));
            stemporal_mean(X_mean, X[jj], m, n_jj);
            ssubtract_mean(X_meanless, X[jj], X_mean, m, n_jj);

            for (ii=0; ii<m; ++ii){
                memcpy(&Y[ii*n_total+n_sum], &X_meanless[ii*n_jj], (n_jj-1)*sizeof(float));
                memcpy(&Z[ii*n_total+n_sum], &X_meanless[ii*n_jj+1], (n_jj-1)*sizeof(float));
            }
            free(X_meanless);
        }
        else {
            for (ii=0; ii<m; ++ii){
                memcpy(&Y[ii*n_total+n_sum], &X[jj][ii*n_jj], (n_jj-1)*sizeof(float));
                memcpy(&Z[ii*n_total+n_sum], &X[jj][ii*n_jj+1], (n_jj-1)*sizeof(float));
            }
        }
        n_sum += n_jj-1;
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

        if (remove_mean){
            X_meanless = (double*)malloc(m*n_jj*sizeof(double));
            dtemporal_mean(X_mean, X[jj], m, n_jj);
            dsubtract_mean(X_meanless, X[jj], X_mean, m, n_jj);

            for (ii=0; ii<m; ++ii){
                memcpy(&Y[ii*n_total+n_sum], &X_meanless[ii*n_jj], (n_jj-1)*sizeof(double));
                memcpy(&Z[ii*n_total+n_sum], &X_meanless[ii*n_jj+1], (n_jj-1)*sizeof(double));
            }
            free(X_meanless);
        }
        else {
            for (ii=0; ii<m; ++ii){
                memcpy(&Y[ii*n_total+n_sum], &X[jj][ii*n_jj], (n_jj-1)*sizeof(double));
                memcpy(&Z[ii*n_total+n_sum], &X[jj][ii*n_jj+1], (n_jj-1)*sizeof(double));
        
            }
        }
        n_sum += n_jj-1;
    }
    free(X_mean);
}

void sseparate(float *Y, float *Z, float *X, const int m, const int n, int remove_mean) {

    int ii;

    float *X_mean;
    float *X_meanless;
    X_mean       = (float*)malloc(m*sizeof(float));

    if (remove_mean){
        X_meanless = (float*)malloc(m*n*sizeof(float));
        stemporal_mean(X_mean, X, m, n);
        ssubtract_mean(X_meanless, X, X_mean, m, n);
        
        for (ii=0; ii<m; ++ii){
            memcpy(&Y[ii*(n-1)], &X_meanless[ii*n], (n-1)*sizeof(float));
            memcpy(&Z[ii*(n-1)], &X_meanless[ii*n+1], (n-1)*sizeof(float));
        }
        free(X_meanless);
    }
    else {
        for (ii=0; ii<m; ++ii){
            memcpy(&Y[ii*(n-1)], &X[ii*n], (n-1)*sizeof(float));
            memcpy(&Z[ii*(n-1)], &X[ii*n+1], (n-1)*sizeof(float));
        }
    }
    free(X_mean);
}

void dseparate(double *Y, double *Z, double *X, const int m, const int n, int remove_mean) {

    int ii;

    double *X_mean;
    double *X_meanless;
    X_mean       = (double*)malloc(m*sizeof(double));

    if (remove_mean){
        X_meanless = (double*)malloc(m*n*sizeof(double));
        dtemporal_mean(X_mean, X, m, n);
        dsubtract_mean(X_meanless, X, X_mean, m, n);
        
        for (ii=0; ii<m; ++ii){
            memcpy(&Y[ii*(n-1)], &X_meanless[ii*n], (n-1)*sizeof(double));
            memcpy(&Z[ii*(n-1)], &X_meanless[ii*n+1], (n-1)*sizeof(double));
        }
        free(X_meanless);
    }
    else {
        for (ii=0; ii<m; ++ii){
            memcpy(&Y[ii*(n-1)], &X[ii*n], (n-1)*sizeof(double));
            memcpy(&Z[ii*(n-1)], &X[ii*n+1], (n-1)*sizeof(double));
        }
    }
    free(X_mean);
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