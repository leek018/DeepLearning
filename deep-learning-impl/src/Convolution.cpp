#include "Convolution.h"
#include "im2col.h"
#include "col2im.h"
#include "Mymath.h"
#include <cstring>
sData<double> Convolution::forward(sData<double> x)
{
    //N,C,H,W
    int N = x->params[0];
    int C = x->params[1];
    int H = x->params[2];
    int W  =x->params[3];
    int KN = kernel->params[0];
    int KH = kernel->params[2];
    int KW = kernel->params[3];
    int OH = (H + 2*pad - KH)/stride + 1;
    int OW = (W + 2*pad - KW)/stride + 1;
   
    //for im2col data
    vector<int> col_params(4);
    col_params[0] = N;
    col_params[1] = 1;
    col_params[2] = KH*KW*C;
    col_params[3] = OH*OW;
    col = make_shared<Data<double>>(col_params);
    double *x_start = x->data;
    double *col_start = col->data;
    int col_offset = KH*KW*C*OH*OW;
    int x_offset = C*H*W;
    for(int i = 0 ; i < N; i++)
    {
        im2col_cpu<double>(x_start,C,H,W,KH,stride,pad,col_start);
        col_start +=col_offset;
        x_start += x_offset;
    }
   

 
    //for output data;
    vector<int> output_params(4);
    output_params[0] = N;
    output_params[1] = KN;
    output_params[2] = OH;
    output_params[3] = OW;
    sData<double> output = make_shared<Data<double>>(output_params);
    
    //matmul
    col_start = col->data;
    double *output_start = output->data;
    int output_offset = KN*OH*OW;
    for(int i = 0 ; i < N; i++)
    {
        Dot(CblasNoTrans,CblasNoTrans,KN,OH*OW,KH*KW*C,1.0,kernel->data,KH*KW*C,col_start,OH*OW,0.0,output_start);
        col_start += col_offset;
        output_start += output_offset;
    }
    
    //bias    
    Plus(output->data, bias->data, N, KN*OH*OW, KN);
    X = x;
    return output;
}

sData<double> Convolution::backward(sData<double> dout)
{
    int N = dout->params[0];
    int C = X->params[1];
    int H = X->params[2];
    int W = X->params[3];
    int KN = dout->params[1];
    int KH = kernel->params[2];
    int KW = kernel->params[3];
    int OH = dout->params[2];
    int OW = dout->params[3];

    //dB
    vector<int> dB_params(4);
    dB_params[0] = 1;
    dB_params[1] = KN;
    dB_params[2] = 1;
    dB_params[3] = 1;
    dB = make_shared<Data<double>>(dB_params);
    vectorSum(dout->data,dB->data,N,KN*OH*OW,KN);
    
    //dK
    dK = make_shared<Data<double>>(kernel->params);
    double *temp = new double[kernel->size];
    double *col_start = col->data;
    double *dout_start = dout->data;
    int col_row = C*KH*KW;
    int col_col = OH*OW;
    int dout_row = KN;
    int dout_col = OH*OW;
    int col_offset = col_row*col_col; //KN*KH*KW*OH*OW
    int dout_offset = dout_row*dout_col; //KN*OH*OW
    for(int i = 0 ; i < N; i++)
    {
        //m: FN , n: KH*KW*C, k: OH*OW
        Dot(CblasNoTrans,CblasTrans,KN,KH*KW*C,OH*OW,1.0,dout_start,OH*OW,col_start,OH*OW,0.0,temp);
        col_start += col_offset;
        dout_start += dout_offset;
        Plus(dK->data,temp,1,col_row,col_row);
    }
    
    delete[] temp;      
    
    //dCol
    sData<double> dCol = make_shared<Data<double>>(col->params);    
    dout_start = dout->data;
    double *dcol_start = dCol->data;
    double *kernel_start = kernel->data;
    
    for(int i = 0 ; i < N; i++)
    {
        // M : KH*KW*C, N : OH*OW, K : KN
        Dot(CblasTrans,CblasNoTrans,col_row,col_col,KN,1.0,kernel_start,col_row,dout_start,dout_col,0,dcol_start);
        dout_start+=dout_offset;
        dcol_start+=col_offset;
    }

    //dX
    sData<double> dX = make_shared<Data<double>>(X->params);
    double *dX_start = dX->data;
    dcol_start = dCol->data;
    int dX_offset = C*H*W;
    for(int i = 0 ; i < N; i++)
    {
        col2im_cpu<double>(dcol_start,C,H,W,KH,stride,pad,dX_start);
        dcol_start +=col_offset;
        dX_start += dX_offset;
    }
    return dX;
}
       


