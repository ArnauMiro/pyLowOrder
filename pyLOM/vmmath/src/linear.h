/*
    Linear operator
*/
#ifdef USE_MKL
#define MKL_Complex8 scomplex_t
#define MKL_Complex16 dcomplex_t
#include "mkl.h"
#endif
// Float version
int slinear_operator(float *U, float *S, float *VT, float *Atilde, float *Y, float *Z, const float r, const int m, const int n);
// DOuble version
int dlinear_operator(double *U, double *S, double *VT, double *Atilde, double *Y, double *Z, const double r, const int m, const int n);