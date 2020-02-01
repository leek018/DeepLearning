#include "Layer.h"
#include "Data.h"

class Dropout:public Layer
{
public:
    double dropout_ratio;
    sData<bool> mask;
    Dropout(double dropout_ratio_):dropout_ratio(dropout_ratio_)
    {}
    virtual sData<double> forward(sData<double> x){return x;}
    sData<double> forward(sData<double> x,bool train_flg);
    virtual sData<double> backward(sData<double> dout);
};
