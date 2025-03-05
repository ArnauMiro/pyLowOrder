/*
	Statistics operations
*/
#include <math.h>
#include "mpi.h"

#include "stats.h"

#define AC_MAT(A,n,i,j) *((A)+(n)*(i)+(j))
#define POW2(x)         ((x)*(x))


float sRMSE(float *A, float *B, const int m, const int n) {
	/*
		Compute the Root Meean Square Error (RMSE) between two
		matrices and return it

		A(m,n), B(m,n)
	*/
	int ii, jj, mg, ng;
	float sum1 = 0., norm1 = 0., sum1g = 0.;
//	float sum2 = 0., norm2 = 0., sum2g = 0.;
	float sum2g = 0.;
	#ifdef USE_OMP
	#pragma omp parallel for private(ii,jj) shared(A,B) firstprivate(m,n)
	#endif
	for(ii = 0; ii < m; ++ii) {
		norm1 = 0.;
//		norm2 = 0.;
		for(jj = 0; jj < n; ++jj){
			norm1 += POW2(AC_MAT(A,n,ii,jj) - AC_MAT(B,n,ii,jj));
//			norm2 += POW2(AC_MAT(A,n,ii,jj));
		}
		sum1 += norm1;
//		sum2 += norm2;
	}
	// Reduce MPI parallel run
	MPI_Allreduce(&sum1,&sum1g,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&m,&mg,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&n,&ng,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	sum2g = (float)(mg*ng);
//	MPI_Allreduce(&sum2,&sum2g,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
	// Return
	return sqrt(sum1g/sum2g);
}

double dRMSE(double *A, double *B, const int m, const int n) {
	/*
		Compute the Root Meean Square Error (RMSE) between two
		matrices and return it

		A(m,n), B(m,n)
	*/
	int ii, jj, mg, ng;
	double sum1 = 0., norm1 = 0., sum1g = 0.;
//	double sum2 = 0., norm2 = 0., sum2g = 0.;
	double sum2g = 0.;
	#ifdef USE_OMP
	#pragma omp parallel for private(ii,jj) shared(A,B) firstprivate(m,n)
	#endif
	for(ii = 0; ii < m; ++ii) {
		norm1 = 0.;
//		norm2 = 0.;
		for(jj = 0; jj < n; ++jj){
			norm1 += POW2(AC_MAT(A,n,ii,jj) - AC_MAT(B,n,ii,jj));
//			norm2 += POW2(AC_MAT(A,n,ii,jj));
		}
		sum1 += norm1;
//		sum2 += norm2;
	}
	// Reduce MPI parallel run
	MPI_Allreduce(&sum1,&sum1g,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
//	MPI_Allreduce(&sum2,&sum2g,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&m,&mg,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&n,&ng,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	sum2g = (double)(mg*ng);
	// Return
	return sqrt(sum1g/sum2g);
}
