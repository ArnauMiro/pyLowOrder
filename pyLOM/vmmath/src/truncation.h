/*
	Truncation operations
*/

int  compute_truncation_residual(double *S, double res, const int n);
void compute_truncation(double *Ur, double *Sr, double *VTr, double *U,	double *S, double *VT, const int m, const int n, const int N);
