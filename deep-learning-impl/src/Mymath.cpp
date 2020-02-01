#include <cmath>
#include <random>
#include <cstring>
#include "Mymath.h"
extern "C"{
    #include <cblas.h>
}
#define abs(X) ( (X) < 0 ? (-(X)) : (X))
using namespace std;
void Dot(const CBLAS_TRANSPOSE TransA,const CBLAS_TRANSPOSE TransB,const int M,const int N,const int K,const double alpha,const double *A,int lda,const double *B,int ldb,const double beta,double *output)
{
    memset(output,0,sizeof(double)*M*N);
    cblas_dgemm(CblasRowMajor,TransA,TransB,M,N,K,alpha,A,lda,B,ldb,beta,output,N);	
}

void Plus(double *A, const double *B,const int A_r,const int A_c,const int B_c)
{
    int inChargeOf = A_c / B_c;
	for (int i = 0; i < A_r; i++)
	{
		for (int j = 0; j < B_c; j++)
		{
            for(int k = 0 ; k < inChargeOf; k++)
                A[(i*B_c+j)*inChargeOf + k] += B[j];
		}
	}
}

double* Transpose(double* target, int r, int c)
{
	double* output = new double[r*c];
	for (int i = 0; i < r; i++)
	{
		for (int j = 0; j < c; j++)
			output[j*r+i] = target[i*c+j];
	}
	return output;
}

void vectorSum(const double *dout,double *dB,const int dout_r,const int dout_c,const int dB_c)
{
    memset(dB,0,sizeof(double)*dB_c);
    int inChargeOf = dout_c/dB_c;
    for(int i = 0 ; i < dout_r; i++)
    {
        for(int j = 0 ; j < dB_c; j++)
        {
            for(int k = 0 ; k < inChargeOf; k++)
            {
                dB[j] += dout[(i*dout_c+j)*inChargeOf+k];
            }
        }
    }	
}

void softMax(const double* input,double *output,const int batch_size,const int output_size)
{
	for (int i = 0; i < batch_size; i++)
	{
		double maxVal = input[i*output_size];
		for (int j = 1; j < output_size; j++)
			maxVal = maxVal < input[i*output_size+j] ? input[i*output_size+j] : maxVal;
		double sum = 0;
		for (int j = 0; j < output_size; j++){
            double e = exp(input[i*output_size+j] - maxVal);
            output[i*output_size+j] = e;
			sum += e;
        }
		for (int j = 0; j < output_size; j++)
			output[i*output_size+j] /= sum;
	}
}

double crossEntropyError(double* y, double* t,int output_size,int batch_size)
{
	double sum = 0;
	for (int i = 0; i < batch_size; i++)
	{
		for (int j = 0; j < output_size; j++)
			sum += log(y[i*output_size+j] + 1e-7) * t[i*output_size+j];
	}

	return -sum / (double)batch_size;
}

void initParams(double *target,int r,int c,double alpha)
{
    random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<> val(0,1.0);
    for(int i = 0 ; i < r*c; i++)
        target[i] = val(gen)*alpha;
}

bool* randomGenerate(int range,int num)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> val(0,range-1);
    int count = 0 ;
    bool *visit = new bool[range];
    memset(visit,0,sizeof(bool)*range);
    while(count !=num)
    {
        int target = val(gen);
        if(!visit[target])
        {
            visit[target] = true;
            count++;
        }
    }
    return visit;
}



int* findMax(double *target,int r,int c)
{    
    int* arr = new int[r];
    for (int i = 0; i < r; i++)
    {
        int MaxIdx = 0;
        double Max = target[i*c];
        for (int j = 1; j < c; j++)
        {
            if (Max < target[i*c+j])
            {
                Max = target[i*c+j];
                MaxIdx = j;
            }
        }
        arr[i] = MaxIdx;
    }
    return arr;
}

double diff(double *a,double *b,const int &size)
{
    double sum = 0;
    for(int i = 0 ; i < size; i++)
        sum +=(abs(a[i]-b[i]));
    return sum;
}
double avg(const double &sum,int size)
{
    return sum/(double)size;
}
   
