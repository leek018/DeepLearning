#ifndef LAYER_H
#define LAYER_H
#include "Data.h"
using namespace std;
class Layer
{
public:
    
    virtual ~Layer(){}
    virtual sData<double> forward(sData<double> x)=0;
    virtual sData<double> backward(sData<double> dout)=0;
};
#endif
