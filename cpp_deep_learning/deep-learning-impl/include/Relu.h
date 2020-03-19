#include "Layer.h"
using namespace std;

class Relu:public Layer
{
public:
	sData<bool> mask;
    virtual	sData<double> forward(sData<double> x);
	virtual sData<double> backward(sData<double> dout);
};
