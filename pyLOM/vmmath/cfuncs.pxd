#!/usr/bin/env cpython
#
# pyLOM - Python Low Order Modeling.
#
# Math operations module - exporting of C functions.
#
# Last rev: 27/10/2021

cimport numpy as np


## Expose C functions
cdef extern from "vector_matrix.h" nogil:
	# Single precision
	cdef void   c_stranspose         "stranspose"(float *A, float *B, const int m, const int n)
	cdef float  c_svector_sum        "svector_sum"(float *v, int start, int n)
	cdef float  c_svector_norm       "svector_norm"(float *v, int start, int n)
	cdef float  c_svector_mean       "svector_mean"(float *v, int start, int n)
	cdef void   c_smatmult           "smatmult"(float *C, float *A, float *B, const int m, const int n, const int k, const char *TA, const char *TB)
	cdef void   c_smatmul            "smatmul"(float *C, float *A, float *B, const int m, const int n, const int k)
	cdef void   c_smatmulp           "smatmulp"(float *C, float *A, float *B, const int m, const int n, const int k)
	cdef void   c_svecmat            "svecmat"(float *v, float *A, const int m, const int n)
	cdef int    c_sinv               "sinv"(float *A, int m, int n)
	cdef int    c_sinverse           "sinverse"(float *A, int N, char *UoL)
	cdef void   c_ssort              "ssort"(float *v, int *index, int n)
	# Double precision
	cdef void   c_dtranspose         "dtranspose"(double *A, double *B, const int m, const int n)
	cdef double c_dvector_sum        "dvector_sum"(double *v, int start, int n)
	cdef double c_dvector_norm       "dvector_norm"(double *v, int start, int n)
	cdef double c_dvector_mean       "dvector_mean"(double *v, int start, int n)
	cdef void   c_dmatmult           "dmatmult"(double *C, double *A, double *B, const int m, const int n, const int k, const char *TA, const char *TB)
	cdef void   c_dmatmul            "dmatmul"(double *C, double *A, double *B, const int m, const int n, const int k)
	cdef void   c_dmatmulp           "dmatmulp"(double *C, double *A, double *B, const int m, const int n, const int k)
	cdef void   c_dvecmat            "dvecmat"(double *v, double *A, const int m, const int n)
	cdef int    c_dinv               "dinv"(double *A, int m, int n)
	cdef int    c_dinverse           "dinverse"(double *A, int N, char *UoL)
	cdef void   c_dsort              "dsort"(double *v, int *index, int n)
	# Single complex precision
	cdef void   c_cmatmult           "cmatmult"(np.complex64_t *C, np.complex64_t *A, np.complex64_t *B, const int m, const int n, const int k, const char *TA, const char *TB)
	cdef void   c_cmatmul            "cmatmul"(np.complex64_t *C, np.complex64_t *A, np.complex64_t *B, const int m, const int n, const int k)
	cdef void 	c_cmatmulp           "cmatmulp"(np.complex64_t *C, np.complex64_t *A, np.complex64_t *B, const int m, const int n, const int k)
	cdef void   c_cvecmat            "cvecmat"(np.complex64_t *v, np.complex64_t *A, const int m, const int n)
	cdef int    c_cinv               "cinv"(np.complex64_t *A, int m, int n)
	cdef int    c_cinverse           "cinverse"(np.complex64_t *A, int N, char *UoL)
	cdef int    c_ceigen             "ceigen"(float *real, float *imag, np.complex64_t *vecs, float *A, const int m, const int n)
	cdef int    c_ccholesky          "ccholesky"(np.complex64_t *A, int N)
	cdef void   c_cvandermonde       "cvandermonde"(np.complex64_t *Vand, float *real, float *imag, int m, int n)
	cdef void   c_cvandermonde_time  "cvandermondeTime"(np.complex64_t *Vand, float *real, float *imag, int m, int n, float* t)
	cdef void   c_csort              "csort"(np.complex64_t *v, int *index, int n)
	# Double complex precision
	cdef void   c_zmatmult           "zmatmult"(np.complex128_t *C, np.complex128_t *A, np.complex128_t *B, const int m, const int n, const int k, const char *TA, const char *TB)
	cdef void   c_zmatmul            "zmatmul"(np.complex128_t *C, np.complex128_t *A, np.complex128_t *B, const int m, const int n, const int k)
	cdef void 	c_zmatmulp           "zmatmulp"(np.complex128_t *C, np.complex128_t *A, np.complex128_t *B, const int m, const int n, const int k)
	cdef void   c_zvecmat            "zvecmat"(np.complex128_t *v, np.complex128_t *A, const int m, const int n)
	cdef int    c_zinv               "zinv"(np.complex128_t *A, int m, int n)
	cdef int    c_zinverse           "zinverse"(np.complex128_t *A, int N, char *UoL)
	cdef int    c_zeigen             "zeigen"(double *real, double *imag, np.complex128_t *vecs, double *A, const int m, const int n)
	cdef int    c_zcholesky          "zcholesky"(np.complex128_t *A, int N)
	cdef void   c_zvandermonde       "zvandermonde"(np.complex128_t *Vand, double *real, double *imag, int m, int n)
	cdef void   c_zvandermonde_time  "zvandermondeTime"(np.complex128_t *Vand, double *real, double *imag, int m, int n, double* t)
	cdef void   c_zsort              "zsort"(np.complex128_t *v, int *index, int n)
cdef extern from "averaging.h" nogil:
	# Single precision
	cdef void c_stemporal_mean       "stemporal_mean"(float *out, float *X, const int m, const int n)
	cdef void c_ssubtract_mean       "ssubtract_mean"(float *out, float *X, float *X_mean, const int m, const int n)
	# Double precision
	cdef void c_dtemporal_mean       "dtemporal_mean"(double *out, double *X, const int m, const int n)
	cdef void c_dsubtract_mean       "dsubtract_mean"(double *out, double *X, double *X_mean, const int m, const int n)
cdef extern from "svd.h" nogil:
	# Single precision
	cdef int c_sqr                   "sqr"(float *Q, float *R, float *A,  const int m, const int n)
	cdef int c_ssvd                  "ssvd"(float *U, float *S, float *V, float *Y, const int m, const int n)
	cdef int c_stsqr                 "stsqr"(float *Qi, float *R, float *Ai, const int m, const int n)
	cdef int c_stsqr_svd             "stsqr_svd"(float *Ui, float *S, float *VT, float *Ai, const int m, const int n)
	cdef int c_srandomized_qr        "srandomized_qr"(float *Qi, float *B, float *Ai, const int m, const int n, const int r, const int q, unsigned int seed)
	cdef int c_sinit_randomized_qr   "sinit_randomized_qr"(float *Qi, float *B, float *Y, float *Ai, const int m, const int n, const int r, const int q, unsigned int seed)
	cdef int c_supdate_randomized_qr "supdate_randomized_qr"(float *Q2, float *B2, float *Yn, float *Q1, float *B1, float *Yo, float *Ai, const int m, const int n, const int n1, const int n2, const int r, const int q, unsigned int seed)
	cdef int c_srandomized_svd       "srandomized_svd"(float *Ui, float *S, float *VT, float *Ai, const int m, const int n, const int r, const int q, unsigned int seed)
	# Double precision
	cdef int c_dqr                   "dqr"(double *Q, double *R, double *A, const int m, const int n)
	cdef int c_dsvd                  "dsvd"(double *U, double *S, double *V, double *Y, const int m, const int n)
	cdef int c_dtsqr                 "dtsqr"(double *Qi, double *R, double *Ai, const int m, const int n)
	cdef int c_dtsqr_svd             "dtsqr_svd"(double *Ui, double *S, double *VT, double *Ai, const int m, const int n)
	cdef int c_drandomized_qr        "drandomized_qr" (double *Qi, double *R, double *Ai, const int m, const int n, const int r, const int q, unsigned int seed)
	cdef int c_dinit_randomized_qr   "dinit_randomized_qr"(double *Qi, double *R, double *Y, double *Ai, const int m, const int n, const int r, const int q, unsigned int seed)
	cdef int c_dupdate_randomized_qr "dupdate_randomized_qr"(double *Q2, double *B2, double *Yn, double *Q1, double *B1, double *Yo, double *Ai, const int m, const int n, const int n1, const int n2, const int r, const int q, unsigned int seed)
	cdef int c_drandomized_svd       "drandomized_svd"(double *Ui, double *S, double *VT, double *Ai,  const int m, const int n, const int r, const int q, unsigned int seed)
	# Single complex precision
	cdef int c_cqr                   "cqr"(np.complex64_t *Q, np.complex64_t *R, np.complex64_t *A, const int m, const int n)
	cdef int c_csvd                  "csvd"(np.complex64_t *U, float *S, np.complex64_t *V, np.complex64_t *Y, const int m, const int n)
	cdef int c_ctsqr                 "ctsqr"(np.complex64_t *Qi, np.complex64_t *R, np.complex64_t *Ai, const int m, const int n)
	cdef int c_ctsqr_svd             "ctsqr_svd"(np.complex64_t *Ui, float *S, np.complex64_t *VT, np.complex64_t *Ai, const int m, const int n)
	# Double complex precision
	cdef int c_zqr                   "zqr"(np.complex128_t *Q,  np.complex128_t *R, np.complex128_t *A,  const int m, const int n)
	cdef int c_zsvd                  "zsvd"(np.complex128_t *U,  double *S, np.complex128_t *V,  np.complex128_t *Y,  const int m, const int n)
	cdef int c_ztsqr                 "ztsqr"(np.complex128_t *Qi, np.complex128_t *R, np.complex128_t *Ai, const int m, const int n)
	cdef int c_ztsqr_svd             "ztsqr_svd"(np.complex128_t *Ui, double *S, np.complex128_t *VT, np.complex128_t *Ai, const int m, const int n)
cdef extern from "stats.h" nogil:
	# Single precision
	cdef float  c_sRMSE              "sRMSE"(float *A, float *B, const int m, const int n)
	cdef float  c_sRMSE_relative     "sRMSE_relative"(float *A, float *B, const int m, const int n)
	cdef float  c_sMAE               "sMAE"(float *A, float *B, const int m, const int n)
	cdef float  c_sr2                "sr2"(float *A, float *B, const int m, const int n)
	cdef void   c_sMRE_array         "sMRE_array"(float *MRE, float *A, float *B, const int m, const int n, const int axis)
	# Double precision
	cdef double c_dRMSE              "dRMSE"(double *A, double *B, const int m, const int n)
	cdef double c_dRMSE_relative     "dRMSE_relative"(double *A, double *B, const int m, const int n)
	cdef double c_dMAE               "dMAE"(double *A, double *B, const int m, const int n)
	cdef double c_dr2                "dr2"(double *A, double *B, const int m, const int n)
	cdef void   c_dMRE_array         "dMRE_array"(double *MRE, double *A, double *B, const int m, const int n, const int axis)
cdef extern from "fft.h" nogil:
	cdef int USE_FFTW3               "_USE_FFTW3"
	# Single precision
	cdef void c_shammwin             "shammwin"(float *out, const int N)
	cdef void c_sfft1D               "sfft1D"(np.complex64_t *out, float *y, const int n)
	cdef void c_sfft                 "sfft"(float *psd, float *y, const float dt, const int n)
	cdef void c_snfft                "snfft"(float *psd, float *t, float* y, const int n)
	# Double precision
	cdef void c_dhammwin             "dhammwin"(double *out, const int N)
	cdef void c_dfft1D               "dfft1D"(np.complex128_t *out, double *y, const int n)
	cdef void c_dfft                 "dfft"(double *psd, double *y, const double dt, const int n)
	cdef void c_dnfft                "dnfft"(double *psd, double *t, double* y, const int n)
cdef extern from "geometric.h" nogil:
	# Single precision
	cdef void c_scellCenters         "scellCenters"(float *xyzc, float *xyz, int *conec, const int nel, const int ndim, const int ncon)
	cdef void c_snormals             "snormals"(float *normals, float *xyz, int *conec, const int nel, const int ndim, const int ncon)
	cdef void c_seuclidean_d         "seuclidean_d"(float *D, float *X, const int m, const int n)
	# Double precision
	cdef void c_dcellCenters         "dcellCenters"(double *xyzc, double *xyz, int *conec, const int nel, const int ndim, const int ncon)
	cdef void c_dnormals             "dnormals"(double *normals, double *xyz, int *conec, const int nel, const int ndim, const int ncon)
	cdef void c_deuclidean_d         "deuclidean_d"(double *D, double *X, const int m, const int n)
cdef extern from "truncation.h" nogil:
	# Single precision
	cdef int    c_scompute_truncation_residual "scompute_truncation_residual"(float *S, float res, const int n)
	cdef void   c_scompute_truncation          "scompute_truncation"(float *Ur, float *Sr, float *VTr, float *U, float *S, float *VT, const int m, const int n, const int nmod, const int N)
	cdef float  c_senergy                      "senergy"(float *A, float *B, const int m, const int n)
	# Double precision
	cdef int    c_dcompute_truncation_residual "dcompute_truncation_residual"(double *S, double res, const int n)
	cdef void   c_dcompute_truncation          "dcompute_truncation"(double *Ur, double *Sr, double *VTr, double *U, double *S, double *VT, const int m, const int n, const int nmod, const int N)
	cdef double c_denergy                      "denergy"(double *A, double *B, const int m, const int n)
cdef extern from "regression.h" nogil:
	# Float version
	cdef void c_sleast_squares    "sleast_squares"(float *out, float *A, float *b, const int m, const int n)
	cdef void c_sridge_regression "sridge_regression"(float *out, float *A, float *b, float lam, const int m, const int n)
	# Double version
	cdef void c_dleast_squares    "dleast_squares"(double *out, double *A, double *b, const int m, const int n)
	cdef void c_dridge_regression "dridge_regression"(double *out, double *A, double *b, double lam, const int m, const int n)


## Fused type between double and complex
ctypedef fused real:
	float
	double
ctypedef fused real_complex:
	np.complex64_t
	np.complex128_t
ctypedef fused real_full:
	float
	double
	np.complex64_t
	np.complex128_t