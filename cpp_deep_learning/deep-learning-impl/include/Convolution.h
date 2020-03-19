#include "Data.h"
#include "Layer.h"
class Convolution:public Layer
{
public:
    sData<double> kernel,bias;
    int stride,pad;
    
    //for backward
    sData<double> X,col;
    
    //derivatives for kernel, bias
    sData<double> dK,dB;
    
    Convolution(int stride_,int pad_,sData<double> &kernel_,sData<double> &bias_):stride(stride_),pad(pad_),kernel(kernel_),bias(bias_){}

    virtual sData<double> forward(sData<double> x);
    virtual sData<double> backward(sData<double> dout);
};
     
