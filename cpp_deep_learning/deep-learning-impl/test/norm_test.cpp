#include <iostream>
#include <vector>
#include "Data.h"
#include "BatchNormalization.h"

using namespace std;

int main()
{
    int N = 5;
    int C = 1;
    int H = 1;
    int W = 5;
    
    vector<int> test_params(4);
    
    test_params[0] = 1;
    test_params[1] = 1;
    test_params[2] = 1;
    test_params[3] = W;
    
    vector<int> running_params(4);

    sData<double> test_gamma = make_shared<Data<double>>(test_params);
    sData<double> test_beta = make_shared<Data<double>>(test_params);

    //init gamma and beta(initially zero)
    fill_n(test_gamma->data,test_gamma->size,1);
    
    sData<double> running_mean = make_shared<Data<double>>(running_params);
    sData<double> running_var = make_shared<Data<double>>(running_params);
    
    double momentum = 0.9;
    Layer *l = new BatchNormalization(test_gamma,test_beta,momentum,running_mean,running_var);
    
    vector<int> input_params(4);
    input_params[0] = N;
    input_params[1] = C;
    input_params[2] = H;
    input_params[3] = W;
    
    sData<double> input = make_shared<Data<double>>(input_params);
    for(int i = 0 ; i < input->size; i++)
        input->data[i] = i;

    bool train_flag = true;
    sData<double> forward_output = dynamic_cast<BatchNormalization*>(l)->forward(input,train_flag);
    cout << "input : \n";
    for(int i = 0 ; i < N; i++)
    {
        for(int j = 0; j < W; j++)
            cout << input->data[i*W+j] << " ";
        cout <<"\n";
    }
    cout <<"forward at TRAIN:\n";
    for(int i = 0 ; i < N; i++)
    {
        for(int j = 0; j < W; j++)
            cout << forward_output->data[i*W+j] << " ";
        cout <<"\n";
    }

    train_flag = false; 
    forward_output = dynamic_cast<BatchNormalization*>(l)->forward(input,train_flag);
    cout <<"forward at TEST:\n";
    for(int i = 0 ; i < N; i++)
    {
        for(int j = 0; j < W; j++)
            cout << forward_output->data[i*W+j] << " ";
        cout <<"\n";
    }
    
    
    sData<double> backward_output = dynamic_cast<BatchNormalization*>(l)->backward(input);
    cout <<"backward :\n";
    for(int i = 0 ; i < N; i++)
    {
        for(int j = 0; j < W; j++)
            cout << backward_output->data[i*W+j] << " ";
        cout <<"\n";
    }
    return 0; 
}
