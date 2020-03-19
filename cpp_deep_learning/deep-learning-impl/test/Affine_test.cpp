#include <iostream>
#include "Affine.h"
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
    int channel = 1;
    int input_size = 2;
    int output_size = 3;
    vector<int> input_param(4);
    vector<int> weight_param(4);
    vector<int> bias_param(4);
    weight_param[0] = 1;
    weight_param[1] = 1;    
    weight_param[2] = input_size;
    weight_param[3] = output_size;
    
    input_param[0] = batch_size;
    input_param[1] = channel;
    input_param[2] = 1;
    input_param[3] = input_size;

    bias_param[0] = 1;
    bias_param[1] = 1;
    bias_param[2] = 1;
    bias_param[3] = output_size;

    sData<double> input = make_shared<Data<double>>(input_param);    
    sData<double> weight = make_shared<Data<double>>(weight_param);
    sData<double> bias = make_shared<Data<double>> (bias_param);

    int total_size = 1;
    for(int i = 0 ; i < input->params.size(); i++)
        total_size *= input->params[i];
    for(int i = 0 ; i < total_size; i++)
        input->data[i] = i+1;
    
    total_size = 1;
    for(int i = 0 ; i < weight->params.size(); i++)
        total_size *= weight->params[i];
    for(int i = 0 ; i < total_size; i++)
            weight->data[i] = i+1;

    total_size = 1;
    for(int i = 0 ; i < bias->params.size(); i++)
        total_size *= bias->params[i];
    for(int i = 0 ; i < total_size; i++)
        bias->data[i] = 1;
   
    cout <<"============forward test===========\n";
    Affine *a = new Affine(weight,bias);
    cout <<"input : \n";
    print(input->data,batch_size,input_size);
    cout <<"weight : \n";
    print(weight->data,input_size,output_size);
    sData<double> output = a->forward(input);
    cout <<"output: \n";
    print(output->data,batch_size,output_size);   

   
    cout <<"============backward test===========\n";
    vector<int> dout_param(4);
    dout_param[0] = batch_size;
    dout_param[1] = 1;
    dout_param[2] = 1;
    dout_param[3] = output_size; 
    sData<double> dout = make_shared<Data<double>>(dout_param);

    total_size = 1;
    for(int i = 0 ; i < dout->params.size(); i++)
        total_size *= dout->params[i];

    for(int i = 0 ; i < total_size; i++)
        dout->data[i] = i+1;

    cout << "total size : " << total_size <<"\n"; 
    cout << "dout: \n";
    print(dout->data,batch_size,output_size);
    sData<double> dX = a->backward(dout);
    
    cout <<"dX : \n";
    print(dX->data,batch_size,input_size);
    cout <<"dW : \n";
    print(a->dW->data,input_size,output_size);
    cout <<"dB : \n";
    print(a->dB->data,1,output_size);
   
    return 0; 

}    
