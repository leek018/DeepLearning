#include "Adam.h"
#include <cmath>
#include <cstring>
using namespace std;
void Adam::update(vector<sData<double>> &params,vector<sData<double>> &grads)
{
    if(m.empty())
    {
        for(int i = 0 ; i < params.size(); i++){                        
            sData<double> &dataholder = params[i];
            m.push_back( make_shared<Data<double>>(dataholder->params) );            
            v.push_back(make_shared<Data<double>>(dataholder->params));
            int total_size = dataholder->size;
            memset(m[i]->data,0,sizeof(double)*total_size);
            memset(v[i]->data,0,sizeof(double)*total_size);
        }
    }
    
    iter++;
    double lr_t = lr* sqrt(1.0-pow(beta2,(double)iter)) / (1.0-pow(beta1,(double)iter));
    for(int i = 0 ; i < params.size(); i++)
    {
        sData<double> &dataholder = params[i];
        int total_size = dataholder->size;
        for(int j = 0 ; j < total_size; j++)
        {
            m[i]->data[j] += (1 - beta1) * (grads[i]->data[j] - m[i]->data[j]);
            v[i]->data[j] += (1 - beta2) * (grads[i]->data[j]*grads[i]->data[j] - v[i]->data[j]);
        }
    
        for(int j = 0 ; j < total_size; j++)
            dataholder->data[j] -= (lr_t*m[i]->data[j])/ sqrt(v[i]->data[j]+1e-7);
    }
}
        
