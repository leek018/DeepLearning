#include "Layer.h"
#include "Data.h"
#include <vector>
class Affine:public Layer
{
public:
	sData<double> W,B,X,dW,dB;
    Affine(sData<double> &Weight,sData<double> &Bias);

	virtual sData<double> forward(sData<double> x);
	virtual sData<double> backward(sData<double> dout);
};

