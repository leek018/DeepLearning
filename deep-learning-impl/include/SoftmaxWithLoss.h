#include "Layer.h"

class SoftmaxWithLoss
{
public:
	double loss;
	sData<double> y, t;
	double forward(sData<double> x,sData<double> t);
	sData<double> backward();
};
