/*
	Geometric and mesh operations
*/
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#include "geometric.h"

#define AC_MAT(A,n,i,j) *((A)+(n)*(i)+(j))


void scellCenters(float *xyzc, float *xyz, int *conec, const int nel, const int ndim, const int ncon) {
	/*
		Compute the cell centers given a list of elements.
	*/
	int ielem, icon, idim, cc, c;

	for(ielem=0; ielem<nel; ++ielem) {
		// Zero centers array
		for(idim=0; idim<ndim; ++idim)
			AC_MAT(xyzc,ndim,ielem,idim) = 0.;
		cc = 0;
		// Get the values of the field and the positions of the element
		for(icon=0; icon<ncon; ++icon) {
			c = AC_MAT(conec,ncon,ielem,icon);
			if (c < 0) break; // Control multi-element
			// Accumulate
			for(idim=0; idim<ndim; ++idim)
				AC_MAT(xyzc,ndim,ielem,idim) += AC_MAT(xyz,ndim,c,idim);
			++cc;
		}
		// Average
		for(idim=0; idim<ndim; ++idim)
			AC_MAT(xyzc,ndim,ielem,idim) /= (float)(cc);
	}
}

void dcellCenters(double *xyzc, double *xyz, int *conec, const int nel, const int ndim, const int ncon) {
	/*
		Compute the cell centers given a list of elements.
	*/
	int ielem, icon, idim, cc, c;

	for(ielem=0; ielem<nel; ++ielem) {
		// Zero centers array
		for(idim=0; idim<ndim; ++idim)
			AC_MAT(xyzc,ndim,ielem,idim) = 0.;
		cc = 0;
		// Get the values of the field and the positions of the element
		for(icon=0; icon<ncon; ++icon) {
			c = AC_MAT(conec,ncon,ielem,icon);
			if (c < 0) break; // Control multi-element
			// Accumulate
			for(idim=0; idim<ndim; ++idim)
				AC_MAT(xyzc,ndim,ielem,idim) += AC_MAT(xyz,ndim,c,idim);
			++cc;
		}
		// Average
		for(idim=0; idim<ndim; ++idim)
			AC_MAT(xyzc,ndim,ielem,idim) /= (double)(cc);
	}
}

void snormals(float *normals, float *xyz, int *conec, const int nel, const int ndim, const int ncon) {
	/*
		Compute the cell centers given a list of elements.
	*/
	int ielem, icon, idim, c, cc;
	float *cen, *u, *v;
	// Allocate memory buffers
	cen = (float*)malloc(ndim*sizeof(float));
	u   = (float*)malloc(3*sizeof(float));
	v   = (float*)malloc(3*sizeof(float));

	for(ielem=0; ielem<nel; ++ielem) {
		// Zero arrays
		for(idim=0; idim<ndim; ++idim)  {
			cen[idim] = 0.;
			AC_MAT(normals,ndim,ielem,idim) = 0.;
		}
		// Compute centroid
		cc = 0;
		for(icon=0; icon<ncon; ++icon) {
			c = AC_MAT(conec,ncon,ielem,icon);
			if (c < 0) break;
			// Accumulate
			for(idim=0; idim<ndim; ++idim)
				cen[idim] += AC_MAT(xyz,ndim,c,idim);
			++cc;
		}
		// Average
		for(idim=0; idim<ndim; ++idim)
			cen[idim] /= (float)(cc);
		// Compute normal
		// Compute u, v
		icon = cc - 1;
		c    = AC_MAT(conec,ncon,ielem,0);
		cc   = AC_MAT(conec,ncon,ielem,icon);
		for(idim=0; idim<ndim; ++idim) {
			u[idim] = AC_MAT(xyz,ndim,c,idim)  - cen[idim];
			v[idim] = AC_MAT(xyz,ndim,cc,idim) - cen[idim];
		}
		// Cross product
		AC_MAT(normals,3,ielem,0) += 0.5*(u[1]*v[2] - u[2]*v[1]);
		AC_MAT(normals,3,ielem,1) += 0.5*(u[2]*v[0] - u[0]*v[2]);
		AC_MAT(normals,3,ielem,2) += 0.5*(u[0]*v[1] - u[1]*v[0]);
		for(icon=1; icon<ncon; ++icon) {
			c  = AC_MAT(conec,ncon,ielem,icon);
			cc = AC_MAT(conec,ncon,ielem,icon-1);
			if (c < 0) break; // Control multi-element
			// Compute u, v
			for(idim=0; idim<ndim; ++idim) {
				u[idim] = AC_MAT(xyz,ndim,c,idim)  - cen[idim];
				v[idim] = AC_MAT(xyz,ndim,cc,idim) - cen[idim];
			}
			// Cross product
			AC_MAT(normals,3,ielem,0) += 0.5*(u[1]*v[2] - u[2]*v[1]);
			AC_MAT(normals,3,ielem,1) += 0.5*(u[2]*v[0] - u[0]*v[2]);
			AC_MAT(normals,3,ielem,2) += 0.5*(u[0]*v[1] - u[1]*v[0]);
		}
	}
	// Free memory buffers
	free(cen);
	free(u);
	free(v);
}

void dnormals(double *normals, double *xyz, int *conec, const int nel, const int ndim, const int ncon) {
	/*
		Compute the cell centers given a list of elements.
	*/
	int ielem, idim, icon, c, cc;
	double *cen, *u, *v;
	// Allocate memory buffers
	cen = (double*)malloc(ndim*sizeof(double));
	u   = (double*)malloc(3*sizeof(double));
	v   = (double*)malloc(3*sizeof(double));

	for(ielem=0; ielem<nel; ++ielem) {
		// Zero arrays
		for(idim=0; idim<ndim; ++idim)  {
			cen[idim] = 0.;
			AC_MAT(normals,ndim,ielem,idim) = 0.;
		}
		// Compute centroid
		cc = 0;
		for(icon=0; icon<ncon; ++icon) {
			c = AC_MAT(conec,ncon,ielem,icon);
			if (c < 0) break;
			// Accumulate
			for(idim=0; idim<ndim; ++idim)
				cen[idim] += AC_MAT(xyz,ndim,c,idim);
			++cc;
		}
		// Average
		for(idim=0; idim<ndim; ++idim)
			cen[idim] /= (double)(cc);
		// Compute normal
		// Compute u, v
		icon = cc - 1;
		c    = AC_MAT(conec,ncon,ielem,0);
		cc   = AC_MAT(conec,ncon,ielem,icon);
		for(idim=0; idim<ndim; ++idim) {
			u[idim] = AC_MAT(xyz,ndim,c,idim)  - cen[idim];
			v[idim] = AC_MAT(xyz,ndim,cc,idim) - cen[idim];
		}
		// Cross product
		AC_MAT(normals,3,ielem,0) += 0.5*(u[1]*v[2] - u[2]*v[1]);
		AC_MAT(normals,3,ielem,1) += 0.5*(u[2]*v[0] - u[0]*v[2]);
		AC_MAT(normals,3,ielem,2) += 0.5*(u[0]*v[1] - u[1]*v[0]);
		for(icon=1; icon<ncon; ++icon) {
			c  = AC_MAT(conec,ncon,ielem,icon);
			cc = AC_MAT(conec,ncon,ielem,icon-1);
			if (c < 0) break; // Control multi-element
			// Compute u, v
			for(idim=0; idim<ndim; ++idim) {
				u[idim] = AC_MAT(xyz,ndim,c,idim)  - cen[idim];
				v[idim] = AC_MAT(xyz,ndim,cc,idim) - cen[idim];
			}
			// Cross product
			AC_MAT(normals,3,ielem,0) += 0.5*(u[1]*v[2] - u[2]*v[1]);
			AC_MAT(normals,3,ielem,1) += 0.5*(u[2]*v[0] - u[0]*v[2]);
			AC_MAT(normals,3,ielem,2) += 0.5*(u[0]*v[1] - u[1]*v[0]);
		}
	}
	// Free memory buffers
	free(cen);
	free(u);
	free(v);
}

void seuclidean_d(float *D, float *X, const int m, const int n){
	/*
		Compute the Euclidean distance matrix

		In:
			- X: MxN Data matrix with N points in the mesh for M simulations
		Returns:
			- D: NxN distance matrix
	*/
	float d, d2, d2G, dG;

	for (int i = 0; i < n; i++) {
		for (int j = i+1; j < n; j++) {
			d2 = 0.;
			// Local sum on the partition
			for (int k = 0; k<m; k++) {
				d = AC_MAT(X,n,k,i) - AC_MAT(X,n,k,j);
				d2 += d*d;
			}
			// Global sum on the partitions
			MPI_Allreduce(&d2,&d2G,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
			dG = sqrt(d2G);
			// Fill output
			AC_MAT(D,n,i,j) = dG;
			AC_MAT(D,n,j,i) = dG;
		}
	}
}

void deuclidean_d(double *D, double *X, const int m, const int n){
	/*
		Compute the Euclidean distance matrix

		In:
			- X: MxN Data matrix with N points in the mesh for M simulations
		Returns:
			- D: NxN distance matrix
	*/
	double d, d2, d2G, dG;

	for (int i = 0; i < n; i++) {
		for (int j = i+1; j < n; j++) {
			d2 = 0.;
			// Local sum on the partition
			for (int k = 0; k<m; k++) {
				d = AC_MAT(X,n,k,i) - AC_MAT(X,n,k,j);
				d2 += d*d;
			}
			// Global sum on the partitions
			MPI_Allreduce(&d2,&d2G,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
			dG = sqrt(d2G);
			// Fill output
			AC_MAT(D,n,i,j) = dG;
			AC_MAT(D,n,j,i) = dG;
		}
	}
}
