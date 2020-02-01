#include "Affine.h"
#include "Relu.h"
#include "SoftmaxWithLoss.h"
#include "Dropout.h"
#include <vector>
#include "Data.h"
using namespace std;
class TwoLayerNet
{
public:
    int is,hs,os;
    bool weight_dec
    vector<sData<double>> params;
    vector<Layer*> layers;
    SoftmaxWithLoss* lastLayer;
    
    TwoLayerNet(int input_size,int hidden_size,int output_size,double weight_init_std);
    ~TwoLayerNet();
    sData<double> predict(sData<double> x,bool train_flg);
    double loss(sData<double> x,sData<double> t,bool train_flg);
    double accuracy(sData<double> x,sData<double> t);
    void gradient(vector<sData<double>> &grad,sData<double> x,sData<double> t);
    void update(vector<sData<double>> &grad,double learningRate);    
};     
