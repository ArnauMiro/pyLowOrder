/*
	Truncation operations
*/
// Single precision
int   scompute_truncation_residual(float *S, float res, const int n);
void  scompute_truncation(float *Ur, float *Sr, float *VTr, float *U, float *S, float *VT, const int m, const int n, const int nmod, const int N);
float senergy(float *A, float *B, const int m, const int n);
// Double precision
int  dcompute_truncation_residual(double *S, double res, const int n);
void dcompute_truncation(double *Ur, double *Sr, double *VTr, double *U, double *S, double *VT, const int m, const int n, const int nmod, const int N);
double denergy(double *A, double *B, const int m, const int n);
