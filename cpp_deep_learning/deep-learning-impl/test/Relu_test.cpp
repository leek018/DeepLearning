#include <iostream>
#include "Relu.h"
#include <vector>
using namespace std;

int main()
{
    int batch_size = 2;
    int input_size = 2;
    Relu *r = new Relu();
    vector<int> x_param(4);
    x_param[0] = batch_size;
    x_param[1] = 1;
    x_param[2] = 1;
    x_param[3] = input_size;
    
    sData<double> x = make_shared<Data<double>>(x_param);
    x->data[0] = 1.0;
    x->data[1] = -0.5;
    x->data[2] = -2.0;
    x->data[3] = 3.0;
    sData<double> out = r->forward(x);
    cout <<"\n";
    cout <<"input : \n";
    for(int i = 0 ; i < 4; i++)
        cout <<x->data[i] <<" ";
    cout <<"\n";
    cout <<"output : \n";
    for(int i = 0 ; i < 4; i++)
        cout << out->data[i] <<" ";
    cout << "\n";
    return 0;
} 

