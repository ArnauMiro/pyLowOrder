/*
	POD - C Functions to compute POD
*/

void compute_temporal_mean(double *out, double *X, const int m, const int n);
void subtract_temporal_mean(double *out, double *X, double *X_mean, const int m, const int n);
void single_value_decomposition(double *U, double *S, double *VT, double *Y, const int m, const int n);
int compute_truncation_residual(double *S, double res, int n);
void compute_svd_truncation(double *U, double *S, double *VT, double *Y, const int m, const int n, const int N);
void compute_power_spectral_density(double *PSD, double *y, const int n);
void compute_power_spectral_density_on_mode(double *PSD, double *V, const int n, const int m, const int transposed);