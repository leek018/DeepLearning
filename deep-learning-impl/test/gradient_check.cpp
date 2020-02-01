#include <iostream>
#include <random>
#include <cstring>
#include "TwoLayerNet.h"
#include "load_mnist.h"
#include "Mymath.h"
using namespace std;

int main()
{
    //t10k-images.idx3-ubyte  t10k-labels.idx1-ubyte  train-images.idx3-ubyte  train-labels.idx1-ubyte 
    random_device rd;
	mt19937 mersenne(rd());

	int input_size = 784;
	int hidden_size = 5;
	int output_size = 10;
	//hyper parameter
	int batch_size = 1;
	int trainSize = 1;
	double learingRate = 0.1;
	int iteration = 1;
	int testSize = 1;

	TwoLayerNet* network = new TwoLayerNet(input_size, hidden_size, output_size, batch_size,0.01);
	uniform_int_distribution<> idx(0, trainSize - 1);
	//data for model 
	double* x_train = ReadMNIST(trainSize, "../data/train-images.idx3-ubyte");
	double* t_train = ReadMNISTLabel(trainSize, "../data/train-labels.idx1-ubyte");
    
    double* x_train_batch = new double[batch_size*input_size];
	double* t_train_batch = new double[batch_size*output_size];

	//test for model
	
    /*
	double* x_test = ReadMNIST(testSize, "../data/t10k-images.idx3-ubyte");
	double* t_test = ReadMNISTLabel(testSize, "../data/t10k-labels.idx1-ubyte");

	double* x_train_batch = new double[batch_size*input_size];
	double* t_train_batch = new double[batch_size*output_size];
	
    for (int i = 0; i < input_size; i++)
	{
		x[0][i] = 0.01*i;
	}
	t[0][0] = 0; t[0][1] = 1;    
    */

    for (int j = 0; j < batch_size; j++)
    {
        int target = idx(mersenne);
        memcpy(&x_train_batch[j*input_size], &x_train[target*input_size], sizeof(double) * input_size);
        memcpy(&t_train_batch[j*output_size], &t_train[target*output_size], sizeof(double) * output_size);
    }

    double *temp_train_input = new double[batch_size*input_size];
    double *temp_test_input = new double[batch_size*output_size];    
    memcpy(temp_train_input,x_train_batch,sizeof(double)*batch_size*input_size);
    memcpy(temp_test_input,t_train_batch,sizeof(double)*batch_size*output_size);

    WeightSet* grad_numerical = network->numerical_gradient(temp_train_input,temp_test_input,input_size,hidden_size,output_size);

    memcpy(temp_train_input,x_train_batch,sizeof(double)*batch_size*input_size);
    memcpy(temp_test_input,t_train_batch,sizeof(double)*batch_size*output_size);

    WeightSet* backprop = network->gradient(temp_train_input,temp_test_input,input_size,hidden_size,output_size);

    
    int size = batch_size*hidden_size;
    double sum = diff(grad_numerical->W1,backprop->W1,size);
    double ag = avg(sum,size); 
    cout <<"W1 : " <<  ag <<"\n";

    size = hidden_size;
    sum = diff(grad_numerical->b1,backprop->b1,size);
    ag = avg(sum,size);
    cout <<"b1 : " << ag <<"\n";
    
    size = hidden_size*output_size;
    sum = diff(grad_numerical->W2,backprop->W2,size);
    ag = avg(sum,size);
    cout <<"W2 : " << ag <<"\n";

    size = output_size;
    sum = diff(grad_numerical->b2,backprop->b2,size);
    ag = avg(sum,size);
    cout <<"b2 : " << ag <<"\n"; 
    
		
	
	delete network;
    delete[] x_train;
    delete[] t_train;
    delete[] x_train_batch;
    delete[] t_train_batch;
    /*
    delete[] x_test;
    delete[] t_test;
    delete[] x_test_batch;
    delete[] t_test_batch;	
    */
	return 0;      
} 
