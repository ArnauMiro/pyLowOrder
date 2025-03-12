/*
	Regression operations
*/
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"

#include "vector_matrix.h"
#include "regression.h"

#define AC_MAT(A,n,i,j) *((A)+(n)*(i)+(j))


void sleast_squares(float *out, float *A, float *b, const int m, const int n) {
	/*
	Least squares regression
	(A^T * A)^-1 * A^T * b

	A(m,n), b(m), out(n)
	*/
	float *aux1, *aux2, *aux3;
	aux1 = (float*)malloc(n*m*sizeof(float));
	aux2 = (float*)malloc(n*n*sizeof(float));
	aux3 = (float*)malloc(n*sizeof(float));

	stranspose(A,aux1,m,n);       // (A^T)        -> aux1(n,m)
	smatmul(aux2,aux1,A,n,n,m);   // (A^T * A)    -> aux2(n,n)
	sinv(aux2,n,n);               // (A^T * A)^-1 -> aux2(n,n)
	smatmul(aux3,aux1,b,n,1,m);   // A^T * b      -> aux3(n)
	free(aux1);
	smatmul(out,aux2,aux3,n,1,n); // (A^T * A)^-1 * A^T * b -> out(n)
	free(aux2);
	free(aux3);
}

void dleast_squares(double *out, double *A, double *b, const int m, const int n) {
	/*
	Least squares regression
	(A^T * A)^-1 * A^T * b

	out(n), A(m,n), b(m)
	*/
	double *aux1, *aux2, *aux3;
	aux1 = (double*)malloc(n*m*sizeof(double));
	aux2 = (double*)malloc(n*n*sizeof(double));
	aux3 = (double*)malloc(n*sizeof(double));

	dtranspose(A,aux1,m,n);       // (A^T)        -> aux1(n,m)
	dmatmul(aux2,aux1,A,n,n,m);   // (A^T * A)    -> aux2(n,n)
	dinv(aux2,n,n);               // (A^T * A)^-1 -> aux2(n,n)
	dmatmul(aux3,aux1,b,n,1,m);   // A^T * b      -> aux3(n)
	free(aux1);
	dmatmul(out,aux2,aux3,n,1,n); // (A^T * A)^-1 * A^T * b -> out(n)
	free(aux2);
	free(aux3);
}

void sridge_regression(float *out, float *A, float *b, float lam, const int m, const int n) {
	/*
	Ridge regression

	out(n), A(m,n), b(m)
	*/
	int ii;
	float *augmented_A, *augmented_b;
	augmented_A = (float*)malloc((m+n)*n*sizeof(float)); //(m+n) x n
	augmented_b = (float*)malloc((m+n)*sizeof(float));   //(m+n) x 1

	// Copy A and b to their augmented counterpart
	memcpy(augmented_A,A,m*n*sizeof(float));
	memcpy(augmented_b,b,m*sizeof(float));

	// Fill bottom part of augmented_A with zeros
	for(ii=m*n;ii<(m+n)*n;++ii)
		augmented_A[ii] = 0.;
	// Finish filling augmented matrices
	lam = sqrt(lam);
	for(ii=m;ii<(m+n);++ii) {
		AC_MAT(augmented_A,n,ii,ii) = lam; // Fill diagonal of augmented_A 
		augmented_b[ii] = 0.; // Fill augmented b with zeros
	}

	// Call least squares regression
	sleast_squares(out,augmented_A,augmented_b,m+n,n);

	free(augmented_A);
	free(augmented_b);
}

void dridge_regression(double *out, double *A, double *b, double lam, const int m, const int n) {
	/*
	Ridge regression

	out(n), A(m,n), b(m)
	*/
	int ii;
	double *augmented_A, *augmented_b;
	augmented_A = (double*)malloc((m+n)*n*sizeof(double)); //(m+n) x n
	augmented_b = (double*)malloc((m+n)*sizeof(double));   //(m+n) x 1

	// Copy A and b to their augmented counterpart
	memcpy(augmented_A,A,m*n*sizeof(double));
	memcpy(augmented_b,b,m*sizeof(double));

	// Fill bottom part of augmented_A with zeros
	for(ii=m*n;ii<(m+n)*n;++ii)
		augmented_A[ii] = 0.;
	// Finish filling augmented matrices
	lam = sqrt(lam);
	for(ii=m;ii<(m+n);++ii) {
		AC_MAT(augmented_A,n,ii,ii-m) = lam; // Fill diagonal of augmented_A 
		augmented_b[ii] = 0.; // Fill augmented b with zeros
	}

	// Call least squares regression
	dleast_squares(out,augmented_A,augmented_b,m+n,n);

	free(augmented_A);
	free(augmented_b);
}