/*
	Geometric and mesh operations
*/
// Float version
void scellCenters(float *xyzc, float *xyz, int *conec, const int nel, const int ndim, const int ncon);
void snormals(float *normals, float *xyz, int *conec, const int nel, const int ndim, const int ncon);
void seuclidean_d(float *D, float *X, const int m, const int n);
// Double version
void dcellCenters(double *xyzc, double *xyz, int *conec, const int nel, const int ndim, const int ncon);
void dnormals(double *normals, double *xyz, int *conec, const int nel, const int ndim, const int ncon);
void deuclidean_d(double *D, double *X, const int m, const int n);
