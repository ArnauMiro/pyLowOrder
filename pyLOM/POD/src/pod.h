/*
	POD - C Functions to compute POD
*/

void compute_temporal_mean(double *out, double *X, const int m, const int n);
void subtract_temporal_mean(double *out, double *X, double *X_mean, const int m, const int n);
void single_value_decomposition(double *U, double *S, double *V, double *Y, const int m, const int n);