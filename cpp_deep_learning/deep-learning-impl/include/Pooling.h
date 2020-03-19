#include "Data.h"
#include "Layer.h"
class Pooling:public Layer
{
public:
    int pool_h;
    int pool_w;
    int stride;
    int pad;
    sData<double> X;
    sData<int> max_indexes;    

    Pooling(int pool_h_, int pool_w_,int stride_,int pad_):pool_h(pool_h_),pool_w(pool_w_),stride(stride_),pad(pad_){}

    virtual sData<double> forward(sData<double> x);
    virtual sData<double> backward(sData<double> dout);
};
