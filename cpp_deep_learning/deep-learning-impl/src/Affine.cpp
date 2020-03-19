#include "Mymath.h"
#include "Affine.h"
#include <iostream>
using namespace std;
//input_size : input_col_num
//output_size : result_col_num
//batch_size : input_row_num
Affine::Affine(sData<double> &Weight,sData<double> &Bias) :W(Weight),B(Bias)
{
    dB = make_shared<Data<double>>(B->params);
    dW = make_shared<Data<double>>(W->params);		
}

sData<double> Affine::forward(sData<double> x)
{
    X = x;
    int XN,XC,XH,XW,HIDDEN;
    XN = x->params[0];
    XC = x->params[1];
    XH = x->params[2];
    XW = x->params[3];    
    HIDDEN = W->params[3];
    
    //output : (1, 1, N, HIDDEN_SIZE)
    vector<int> output_params(4);    
    output_params[0] = XN;
    output_params[1] = 1;
    output_params[2] = 1;
    output_params[3] = HIDDEN;
    sData<double> output = make_shared<Data<double>>(output_params);
    
    // DOT : (N , C*H*W) * (C*H*W, HIDDEN)
    int m = XN;
    int n = HIDDEN;
    int k = XC*XH*XW;
    Dot(CblasNoTrans,CblasNoTrans,m,n,k,1.0,x->data,k,W->data,n,0.0,output->data);
    Plus(output->data, B->data,m, n,n);
    return output;
}

sData<double> Affine::backward(sData<double> dout)
{
    sData<double> dX = make_shared<Data<double>>(X->params);
    int XN,XC,XH,XW,HIDDEN;
    XN = X->params[0];
    XC = X->params[1];
    XH = X->params[2];
    XW = X->params[3];    
    HIDDEN = W->params[3];
    
    int m = XN;
    int n = XC*XH*XW;
    int k = HIDDEN;
    
    Dot(CblasNoTrans,CblasTrans,m,n,k,1.0,dout->data,k,W->data,k,0.0,dX->data);
    Dot(CblasTrans,CblasNoTrans,n,k,m,1.0,X->data,n,dout->data,k,0.0,dW->data);
    vectorSum(dout->data, dB->data, m, k,k);
    return dX;
}
