#ifndef OPTIMIZER_
#define OPTIMIZER_
#include "Data.h"
#include <vector>
class Optimizer{
public:
    virtual ~Optimizer(){}
    virtual void update(vector<sData<double>> &params,vector<sData<double>> &grad) = 0;
};
#endif
