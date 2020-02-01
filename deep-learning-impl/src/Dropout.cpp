#include "Dropout.h"
#include "Mymath.h"
#include <cstring>
using namespace std;
sData<double> Dropout::forward(sData<double> x,bool train_flg)
{    
    int total_size = x->size;
    sData<double> output = make_shared<Data<double>>(x->params);
    memcpy(output->data,x->data,sizeof(double)*total_size);
    if(train_flg)//implemented as if batch size is not important
    {
        mask = make_shared<Data<bool>>(x->params);
        double *temp = new double[total_size];
        initParams(temp,1,total_size,1);
        for(int i = 0 ; i < total_size; i++)
            mask->data[i] = temp[i] > dropout_ratio ? true : false;
        delete[] temp;
        for(int i = 0 ; i < total_size; i++)
        {
            if(!mask->data[i])
                output->data[i] = 0;
        }        
    }
    else{
        for(int i = 0 ; i < total_size; i++)
            output->data[i] *= (1.0 - dropout_ratio);
    }
    return output;
}

sData<double> Dropout::backward(sData<double> dout)
{
    int total_size = dout->size;
    sData<double> output = make_shared<Data<double>>(dout->params);
    memcpy(output->data,dout->data,sizeof(double)*total_size);
    for(int i = 0 ; i < total_size; i++)
    {
        if(!(mask->data[i]))
            output->data[i] = 0;
    }
    return output;
}

