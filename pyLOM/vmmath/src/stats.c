/*
	Statistics operations
*/
#include <math.h>
#include "mpi.h"

#include "stats.h"

#define AC_MAT(A,n,i,j) *((A)+(n)*(i)+(j))
#define POW2(x)         ((x)*(x))


float sRMSE_relative(float *A, float *B, const int m, const int n) {
	/*
		Compute the Root Mean Square Error (RMSE) between two
		matrices and return it

		A(m,n), B(m,n)
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
	return sqrt(sum1g/sum2g);
}

double dRMSE_relative(double *A, double *B, const int m, const int n) {
	/*
		Compute the Root Mean Square Error (RMSE) between two
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
	return sqrt(sum1g/sum2g);
}

float sRMSE(float *A, float *B, const int m, const int n) {
	/*
		Compute the Root Mean Square Error (RMSE) between two
		matrices and return it

		A(m,n), B(m,n)
	*/
	int ii, jj, mg, ng;
	float sum1 = 0., norm1 = 0., sum1g = 0.;
	float sum2g = 0.;
	#ifdef USE_OMP
	#pragma omp parallel for private(ii,jj) shared(A,B) firstprivate(m,n)
	#endif
	for(ii = 0; ii < m; ++ii) {
		norm1 = 0.;
		for(jj = 0; jj < n; ++jj){
			norm1 += POW2(AC_MAT(A,n,ii,jj) - AC_MAT(B,n,ii,jj));
		}
		sum1 += norm1;
	}
	// Reduce MPI parallel run
	MPI_Allreduce(&sum1,&sum1g,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&m,&mg,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&n,&ng,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	sum2g = (float)(mg*ng);
	// Return
	return sqrt(sum1g/sum2g);
}

double dRMSE(double *A, double *B, const int m, const int n) {
	/*
		Compute the Root Mean Square Error (RMSE) between two
		matrices and return it

		A(m,n), B(m,n)
	*/
	int ii, jj, mg, ng;
	double sum1 = 0., norm1 = 0., sum1g = 0.;
	double sum2g = 0.;
	#ifdef USE_OMP
	#pragma omp parallel for private(ii,jj) shared(A,B) firstprivate(m,n)
	#endif
	for(ii = 0; ii < m; ++ii) {
		norm1 = 0.;
		for(jj = 0; jj < n; ++jj){
			norm1 += POW2(AC_MAT(A,n,ii,jj) - AC_MAT(B,n,ii,jj));
		}
		sum1 += norm1;
	}
	// Reduce MPI parallel run
	MPI_Allreduce(&sum1,&sum1g,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&m,&mg,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&n,&ng,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	sum2g = (double)(mg*ng);
	// Return
	return sqrt(sum1g/sum2g);
}

float sMAE(float *A, float *B, const int m, const int n) {
	/*
		Compute the Mean Absolute Error (MAE) between two
		matrices and return it

		A(m,n), B(m,n)
	*/
	int ii, jj, mg, ng;
	float sum1 = 0., sum1g = 0., sum2g = 0.;
	#ifdef USE_OMP
	#pragma omp parallel for private(ii,jj) shared(A,B) firstprivate(m,n)
	#endif
	for(ii = 0; ii < m; ++ii) 
		for(jj = 0; jj < n; ++jj)
			sum1 += fabs(AC_MAT(A,n,ii,jj) - AC_MAT(B,n,ii,jj));
	// Reduce MPI parallel run
	MPI_Allreduce(&sum1,&sum1g,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&m,&mg,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&n,&ng,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	sum2g = (float)(mg*ng);
	// Return
	return sum1g/sum2g;
}

double dMAE(double *A, double *B, const int m, const int n) {
	/*
		Compute the Mean Absolute Error (MAE) between two
		matrices and return it

		A(m,n), B(m,n)
	*/
	int ii, jj, mg, ng;
	double sum1 = 0., sum1g = 0., sum2g = 0.;
	#ifdef USE_OMP
	#pragma omp parallel for private(ii,jj) shared(A,B) firstprivate(m,n)
	#endif
	for(ii = 0; ii < m; ++ii) 
		for(jj = 0; jj < n; ++jj)
			sum1 += fabs(AC_MAT(A,n,ii,jj) - AC_MAT(B,n,ii,jj));
	// Reduce MPI parallel run
	MPI_Allreduce(&sum1,&sum1g,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&m,&mg,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&n,&ng,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	sum2g = (double)(mg*ng);
	// Return
	return sum1g/sum2g;
}

float sr2(float *A, float *B, const int m, const int n) {
	/*
		Compute the R2 correlation coefficient between two
		matrices and return it

		A(m,n), B(m,n)
	*/
	int ii, jj, mg, ng;
	float num = 0., den = 0., sum = 0., numg = 0., deng = 0., sumg = 0.;
	// Compute the numerator
	#ifdef USE_OMP
	#pragma omp parallel for private(ii,jj) shared(A,B) firstprivate(m,n)
	#endif
	for(ii = 0; ii < m; ++ii) {
		for(jj = 0; jj < n; ++jj) {
			num += POW2(AC_MAT(A,n,ii,jj) - AC_MAT(B,n,ii,jj));
			sum += AC_MAT(A,n,ii,jj);
		}
	}
	// Reduce MPI parallel run
	MPI_Allreduce(&num,&numg,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&sum,&sumg,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&m,&mg,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&n,&ng,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	sumg /= (float)(mg*ng); // Sum now contains the average
	// Compute the denominator
	#ifdef USE_OMP
	#pragma omp parallel for private(ii,jj) shared(A,B) firstprivate(m,n)
	#endif
	for(ii = 0; ii < m; ++ii)
		for(jj = 0; jj < n; ++jj)
			den += POW2(AC_MAT(A,n,ii,jj) - sumg);
	MPI_Allreduce(&den,&deng,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
	// Return
	return 1. - numg/deng;
}

double dr2(double *A, double *B, const int m, const int n) {
	/*
		Compute the R2 correlation coefficient between two
		matrices and return it

		A(m,n), B(m,n)
	*/
	int ii, jj, mg, ng;
	double num = 0., den = 0., sum = 0., numg = 0., deng = 0., sumg = 0.;
	// Compute the numerator
	#ifdef USE_OMP
	#pragma omp parallel for private(ii,jj) shared(A,B) firstprivate(m,n)
	#endif
	for(ii = 0; ii < m; ++ii) {
		for(jj = 0; jj < n; ++jj) {
			num += POW2(AC_MAT(A,n,ii,jj) - AC_MAT(B,n,ii,jj));
			sum += AC_MAT(A,n,ii,jj);
		}
	}
	// Reduce MPI parallel run
	MPI_Allreduce(&num,&numg,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&sum,&sumg,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&m,&mg,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&n,&ng,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	sumg /= (double)(mg*ng); // Sum now contains the average
	// Compute the denominator
	#ifdef USE_OMP
	#pragma omp parallel for private(ii,jj) shared(A,B) firstprivate(m,n)
	#endif
	for(ii = 0; ii < m; ++ii)
		for(jj = 0; jj < n; ++jj)
			den += POW2(AC_MAT(A,n,ii,jj) - sumg);
	MPI_Allreduce(&den,&deng,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	// Return
	return 1. - numg/deng;
}