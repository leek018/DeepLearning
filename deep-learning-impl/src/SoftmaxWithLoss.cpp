#include <iostream>
#include "Mymath.h"
#include "SoftmaxWithLoss.h"
using namespace std;


double SoftmaxWithLoss::forward(sData<double> x, sData<double> t_)
{
    t = t_;
    int H = x->params[0]; //batch size
    int W = x->params[3]; //label size
    y = make_shared<Data<double>>(x->params);
    softMax(x->data,y->data, H, W);
    loss = crossEntropyError(y->data, t->data, W, H);
    return loss;
}

sData<double> SoftmaxWithLoss::backward()
{
    sData<double> dX = make_shared<Data<double>>(t->params);
    int H = t->params[0];
    int W = t->params[3];
    int total_size = H*W;
    for (int i = 0; i < total_size; i++)   
        dX->data[i] = (y->data[i] - t->data[i])/H;   
    return dX;
}
