/*
	Averaging routines
*/
void temporal_mean(double *out, double *X, const int m, const int n);
void subtract_mean(double *out, double *X, double *X_mean, const int m, const int n);