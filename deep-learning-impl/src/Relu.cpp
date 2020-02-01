#include <iostream>
#include <cstring>
#include "Relu.h"
using namespace std;

sData<double> Relu::forward(sData<double> x)
{
    mask = make_shared<Data<bool>>(x->params);
    sData<double> output = make_shared<Data<double>>(x->params);

    int total_size = x->size;
    memset(mask->data,0,sizeof(bool)*total_size);	
    memcpy(output->data, x->data, sizeof(double)*total_size);
    for (int i = 0; i < total_size; i++)
    {
        if (output->data[i] < 0) {
            output->data[i] = 0;
            mask->data[i] = true;
        }					
    }
    return output;
}

sData<double> Relu::backward(sData<double> dout)
{
    int total_size = dout->size;
    for (int i = 0; i < total_size; i++)
    {
        if (mask->data[i])
            dout->data[i] = 0;				
    }
    return dout;
}
