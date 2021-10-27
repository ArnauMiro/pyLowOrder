/*
	Vector and matrix math operations
*/
void   transpose(double *A, const int m, const int n);
double vector_norm(double *v, int start, int n);
void   reorder(double *A, int m, int n, int N);
void   matmul(double *C, double *A, double *B, const int m, const int n, const int k);
void   vecmat(double *v, double *A, const int m, const int n);