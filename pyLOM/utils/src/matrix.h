/*
	Matrix operations
*/

void transpose(double *A, const int m, const int n, const int bsz);
double compute_norm(double *A, int start, int n);
void reorder_matrix(double *A, int m, int n, int N);
