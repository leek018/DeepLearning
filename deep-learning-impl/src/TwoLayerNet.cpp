#include <iostream>
#include "TwoLayerNet.h"
#include "Mymath.h"
#include <cstring>
using namespace std;
TwoLayerNet::TwoLayerNet(int input_size,int hidden_size,int output_size,double weight_init_std):is(input_size),hs(hidden_size),os(output_size)
{
    //W1 : INPUT*HIDDEN, W2 : HIDDEN*OUTPUT, B1 : 1*HIDDEN, B2 : 1*OUTPUT 
    params.push_back(make_shared<Data<double>>(input_size,hidden_size));    
    params.push_back(make_shared<Data<double>>(hidden_size,output_size));
    params.push_back(make_shared<Data<double>>(1,hidden_size));
    params.push_back(make_shared<Data<double>>(1,output_size));

    initParams(params[0]->data,input_size,hidden_size,weight_init_std);     
    initParams(params[1]->data,hidden_size,output_size,weight_init_std);
    //fill_n(W1,input_size*hidden_size,0.1*weight_init_std);    
    //fill_n(W2,output_size*hidden_size,0.1*weight_init_std);    
    memset(params[2]->data,0,sizeof(double)*hidden_size);
    memset(params[3]->data,0,sizeof(double)*output_size);
      
 
    layers.push_back(new Affine(params[0],params[2]));
    layers.push_back(new Relu());
    layers.push_back(new Dropout(0.5)); 
    layers.push_back(new Affine(params[1],params[3]));

    lastLayer = new SoftmaxWithLoss();
    
    
}
TwoLayerNet::~TwoLayerNet()
{
    for(int i = 0 ; i < layers.size(); i++){
        delete layers[i];
        layers[i] = NULL;
    }    
    delete lastLayer;
    lastLayer = NULL;
}   
void print(double *target,int r, int c)
{   
    for(int i = 0 ; i < r; i++)
    {
        for(int j = 0 ; j < c; j++)
        {
            cout << target[i*c+j] <<" ";
        }
        cout <<"\n";
    }
}
  
sData<double> TwoLayerNet::predict(sData<double> x,bool train_flg)
{
    for(int i = 0 ; i < layers.size(); i++)
    {
        if( i == 2)
            x = dynamic_cast<Dropout*>(layers[i])->forward(x,train_flg);
        else
            x = layers[i]->forward(x);
    }            
    return x;
}
double TwoLayerNet::loss(sData<double> x,sData<double> t,bool train_flg)
{
    sData<double> y = predict(x,train_flg);
    return lastLayer->forward(y,t);
}
    
double TwoLayerNet::accuracy(sData<double> x,sData<double> t)
{
    sData<double> y = predict(x,false);
    int *arrA = findMax(y->data,y->r,y->c);
    int *arrB = findMax(t->data,t->r,t->c);
    int count =0;
    for(int i = 0 ; i < y->r ;i++)
    {    
        if(arrA[i] == arrB[i])
            count++;
    }
    delete[] arrA;
    delete[] arrB;
    return (double)count/y->r;
}

void TwoLayerNet::gradient(vector<sData<double>> &grad,sData<double> x,sData<double> t)
{
    
    loss(x,t,true);
    sData<double> dout = lastLayer->backward();
    
    for(int i = layers.size()-1 ; i >=0; i--)
        dout = layers[i]->backward(dout);
  
    if(grad.empty()){ 
        grad.push_back(static_cast<Affine*>(layers[0])->dW);
        grad.push_back(static_cast<Affine*>(layers[3])->dW);
        grad.push_back(static_cast<Affine*>(layers[0])->dB);
        grad.push_back(static_cast<Affine*>(layers[3])->dB);
    } 
    
}

void TwoLayerNet::update(vector<sData<double>> &grad,double learningRate)
{

    for(int i = 0 ; i < params.size(); i++)
    {
        int total_size = params[i]->r * params[i]->c;
        for(int j = 0 ; j < total_size; j++)
        {
            params[i]->data[j] -= learningRate*grad[i]->data[j];
        }
    }
    
}
