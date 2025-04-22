/*
	Averaging routines
*/
// Floats
void stemporal_mean(float *out, float *X, const int m, const int n);
void ssubtract_mean(float *out, float *X, float *X_mean, const int m, const int n);
void stemporal_variance(float *out, float *X, float *X_mean, const int m, const int n);
void snorm_variance(float *out, float *Xnomean, float *X_var, const int m, const int n);
// Double
void dtemporal_mean(double *out, double *X, const int m, const int n);
void dsubtract_mean(double *out, double *X, double *X_mean, const int m, const int n);
void dtemporal_variance(double *out, double *X, double *X_mean, const int m, const int n);
void dnorm_variance(double *out, double *Xnomean, double *X_var, const int m, const int n);