#ifndef MYMATH_DEF
#define MYMATH_DEF
using namespace std;
extern "C"{
    #include <cblas.h>
}
void Dot(const CBLAS_TRANSPOSE TransA,const CBLAS_TRANSPOSE TransB,const int M,const int N,const int K,const double alpha,const double *A,const int lda,const double *B,const int ldb,const double beta,double *output);


void Plus(double *A, const double *B,const int A_r,const int A_c,const int B_c);
double* Transpose(double* target, int r, int c);

void vectorSum(const double *dout,double *dB,const int dout_r,const int dout_c,const int dB_c);

void softMax(const double *input,double* output,const int batch_size,const int output_size);
double crossEntropyError(double *y, double *t,int output_size,int batch_size);
void initParams(double *target,int r, int c,double alpha);
int *findMax(double *target,int r, int c);
double diff(double *a,double *b,const int &size);
double avg(const double &sum,int size);
bool* randomGenerate(int ragne,int num);
#endif
