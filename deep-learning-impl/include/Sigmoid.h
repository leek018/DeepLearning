#include "Layer.h" 
class Sigmoid: public Layer 
{
public:
	sData<double> output;
	virtual sData<double> forward(sData<double> x);
	virtual sData<double> backward(sData<double> dout);
};
