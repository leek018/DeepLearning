#include "Sigmoid.h"
#include <cmath>
#include <iostream>
using namespace std;

sData<double> Sigmoid::forward(sData<double> x)
{    
    output = make_shared<Data<double>>(x->params);
    int H = x->params[0]; //possibly batch size
    int W = x->params[3]; //possibly HIDDEN SIZE
    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)			
            output->data[i*W+j] = (double)1 / (1 + exp(-(x->data[i*W+j])));			
    }
    return output;
}

sData<double> Sigmoid::backward(sData<double> dout)
{
    sData<double> dX = make_shared<Data<double>>(dout->params);
    int H = dout->params[0];
    int W = dout->params[3];
    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; i++)
        {
            dX->data[i*W+j] = dout->data[i*W+j] * output->data[i*W+j] * ((double)1 - (output->data[i*W+j]));
        }
    }
    return dX;
}
