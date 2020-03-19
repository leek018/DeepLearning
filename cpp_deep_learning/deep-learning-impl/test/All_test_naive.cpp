#include <iostream>
#include <vector>
#include "Layer.h"
#include "Affine.h"
#include "Relu.h"
#include "SoftmaxWithLoss.h"
using namespace std;
void print(double *x,int r,int c)
{
    for(int i = 0 ; i < r; i++)
    {
        for(int j = 0 ; j < c; j++)
            cout <<x[i*c+j] <<" ";
        cout <<"\n";
    }
}

int main()
{
    int batch_size = 2;
    int input_size = 2;
    int hidden_size = 3;
    int output_size = 3;

    double x_[] = {1,2,3,4};
    double *x = x_;
    double w1[] = {1,2,3,4,5,6};
    double b1[] = {0,0,0};
    
    double w2[] = {1,2,3,4,5,6,7,8,9};
    double b2[] = {0,0,0};
    
    double t[] = {1,2,3};
    
    /*
    vector<Layer*> layers;
    layers.push_back( new Affine(input_size,hidden_size,batch_size,w1,b1));
    layers.push_back( new Relu(batch_size,hidden_size));
    layers.push_back( new Affine(hidden_size,output_size,batch_size,w2,b2));
    SoftmaxWithLoss *d = new SoftmaxWithLoss(batch_size,output_size);
   
   
    cout <<"Affine input : \n";
    print(x,batch_size,input_size);
    cout <<"w1 weight : \n";
    print(w1,input_size,hidden_size);
    x = static_cast<Affine*>(layers[0])->forward(x);
    cout <<"Affine1 output: \n";
    print(x,batch_size,hidden_size);    
    
  
    x = static_cast<Relu*>(layers[1])->forward(x);
    cout << "Relu output : \n";
    print(x,batch_size,hidden_size);
    
  
    cout <<"Affine2 input : \n";
    print(x,batch_size,hidden_size);
    cout <<"w2 weight : \n";
    print(w2,hidden_size,output_size);
    x = static_cast<Affine*>(layers[2])->forward(x);
    cout <<"Affine2 output: \n";
    print(x,batch_size,output_size);
 
    cout <<"==============================Backward=======================\n";   
    d->forward(x,t);
    double *dout =  d->backward();
    cout <<"softmax dout : \n";
    print(dout,batch_size,output_size);
    
    dout = layers[2]->backward(dout);
    cout <<"Affine2 dX \n";
    print(dout,batch_size,hidden_size);
    
    double *Affine2_dW = static_cast<Affine*>(layers[2])->dW;
    cout <<"Affine2_dW \n";
    print(Affine2_dW,hidden_size,output_size);
    
    double *Affine2_dB = static_cast<Affine*>(layers[2])->dB;
    cout <<"Affine2_dB \n";
    print(Affine2_dB,1,output_size);
    
    cout <<"Affine2_dout \n";    
    print(dout,batch_size,hidden_size);
    
    dout = static_cast<Affine*>(layers[1])->backward(dout);
    cout <<"Relu dout \n";
    print(dout,batch_size,hidden_size);

    dout = static_cast<Affine*>(layers[0])->backward(dout);
    cout <<"Affine1 dX \n";
    print(dout,batch_size,input_size);
    
    double *Affine1_dW = static_cast<Affine*>(layers[0])->dW;
    cout <<"Affine1_dW \n";
    print(Affine1_dW,input_size,hidden_size);
    
    double *Affine1_dB = static_cast<Affine*>(layers[0])->dB;
    cout <<"Affine1_dB \n";
    print(Affine1_dB,1,hidden_size);
    
    cout <<"Affine1_dout \n";    
    print(dout,batch_size,hidden_size);
    */

    /* 
    for(int i = 0 ; i < layers.size(); i++)
    {
        x_ = layers[i]->forward(x_);
    }
    cout <<"Affine output: \n";
    print(x_,batch_size,output_size);

    d->forward(x_,t);
    double* dout = d->backward();

    cout <<"===============backward=============\n";
    
    for(int i = layers.size()-1; i >=0; i--)
        dout = layers[i]->backward(dout);

    cout <<"dout output: \n";
    print(dout,batch_size,hidden_size);    
    */

    
    Affine *a = new Affine(input_size,hidden_size,batch_size,w1,b1);
    cout <<"Affine input : \n";
    print(x,batch_size,input_size);
    cout <<"w1 weight : \n";
    print(w1,input_size,hidden_size);
    double* Affine1_output = a->forward(x);
    cout <<"Affine1 output: \n";
    print(Affine1_output,batch_size,hidden_size);    
    
  
    Relu* r = new Relu(batch_size,hidden_size);
    double *Relu_output = r->forward(Affine1_output);
    cout << "Relu output : \n";
    print(Relu_output,batch_size,hidden_size);
    
    Affine* c = new Affine(hidden_size,output_size,batch_size,w2,b2);
    cout <<"Affine2 input : \n";
    print(Relu_output,batch_size,hidden_size);
    cout <<"w2 weight : \n";
    print(w2,hidden_size,output_size);
    double* Affine2_output = c->forward(Relu_output);
    cout <<"Affine2 output: \n";
    print(Affine2_output,batch_size,output_size);
 
    cout <<"==============================Backward=======================\n";   
    SoftmaxWithLoss *d = new SoftmaxWithLoss(batch_size,output_size);
    d->forward(Affine2_output,t);
    double *softdX =  d->backward();
    cout <<"softmax dout : \n";
    print(softdX,batch_size,output_size);
    
    double *Affine2_dX = c->backward(softdX);
    cout <<"Affine2 dX \n";
    print(Affine2_dX,batch_size,hidden_size);
    
    double *Affine2_dW = c->dW;
    cout <<"Affine2_dW \n";
    print(Affine2_dW,hidden_size,output_size);
    
    double *Affine2_dB = c->dB;
    cout <<"Affine2_dB \n";
    print(Affine2_dB,1,output_size);
    
    cout <<"Affine2_dout \n";    
    print(Affine2_dX,batch_size,hidden_size);
    
    double *Relu_dout = r->backward(Affine2_dX);
    cout <<"Relu dout \n";
    print(Relu_dout,batch_size,hidden_size);

    double *Affine1_dX = a->backward(Relu_dout);
    cout <<"Affine1 dX \n";
    print(Affine1_dX,batch_size,input_size);
    
    double *Affine1_dW = a->dW;
    cout <<"Affine1_dW \n";
    print(Affine1_dW,input_size,hidden_size);
    
    double *Affine1_dB = a->dB;
    cout <<"Affine1_dB \n";
    print(Affine1_dB,1,hidden_size);
    
    cout <<"Affine1_dout \n";    
    print(Affine1_dX,batch_size,hidden_size);
    
       
    return 0; 

}
