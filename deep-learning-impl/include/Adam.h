#include "Optimizer.h"
class Adam:public Optimizer
{
public:
    int iter;
    double lr,beta1,beta2;
    vector<sData<double>> m,v;
    Adam(double lr_,double beta1_,double beta2_):lr(lr_),beta1(beta1_),beta2(beta2_),iter(0){}
    virtual void update(vector<sData<double>> &params,vector<sData<double>> &grad);
};
