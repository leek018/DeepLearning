#include "Layer.h"
class BatchNormalization:public Layer
{
public:
    sData<double> gamma;
    sData<double> beta;
    double momentum;
    
    sData<double> running_mean;
    sData<double> running_var;
    
    sData<double> xc;
    sData<double> std;
    sData<double> xhat;

    //backward data
    sData<double> dgamma;
    sData<double> dbeta;
   
    BatchNormalization(sData<double> &gamma_,sData<double> &beta_,double &momentum_,sData<double> &running_mean_,sData<double> &running_var_)
    :gamma(gamma_),beta(beta_),momentum(momentum_),running_mean(running_mean_),running_var(running_var_){} 
    virtual sData<double> forward(sData<double> x){return x;}
    sData<double> forward(sData<double> x,bool train_flag);
    virtual sData<double> backward(sData<double> dout);
};
