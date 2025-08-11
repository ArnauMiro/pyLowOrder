/*
	Regression operations
*/
// Float version
void sleast_squares(float *out, float *A, float *b, const int m, const int n);
void sridge_regression(float *out, float *A, float *b, float lam, const int m, const int n);
// Double version
void dleast_squares(double *out, double *A, double *b, const int m, const int n);
void dridge_regression(double *out, double *A, double *b, double lam, const int m, const int n);