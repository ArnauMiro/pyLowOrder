/*
	Statistics operations
*/
// Float version
float sRMSE(float *A, float *B, const int m, const int n);
float sRMSE_relative(float *A, float *B, const int m, const int n);
float sMAE(float *A, float *B, const int m, const int n);
float sr2(float *A, float *B, const int m, const int n);
void  sMRE_array(float *MRE, float *A, float *B, const int m, const int n, const int axis);
// Double version
double dRMSE(double *A, double *B, const int m, const int n);
double dRMSE_relative(double *A, double *B, const int m, const int n);
double dMAE(double *A, double *B, const int m, const int n);
double dr2(double *A, double *B, const int m, const int n);
void   dMRE_array(double *MRE, double *A, double *B, const int m, const int n, const int axis);