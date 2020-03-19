#include <iostream>
#include <random>
#include <cstring>
#include "TwoLayerNet.h"
#include "load_mnist.h"
#include "Mymath.h"
#include "Data.h"
#include "Adam.h"
using namespace std;
void print(double *target,int r,int c)
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
int main()
{
    //t10k-images.idx3-ubyte  t10k-labels.idx1-ubyte  train-images.idx3-ubyte  train-labels.idx1-ubyte 
    random_device rd;
	mt19937 mersenne(rd());

	int input_size = 784;
	int hidden_size = 50;
	int output_size = 10;
	//hyper parameter
	int batch_size = 100;
	int trainSize = 60000;
	double lRate = 0.1;
	int iteration = 10000;
	int testSize = 10000;
    
    //cin >> iteration;    




	TwoLayerNet* network = new TwoLayerNet(input_size, hidden_size, output_size,0.01);
    Optimizer *optimizer = dynamic_cast<Adam*>(new Adam(0.001,0.9,0.999));
	uniform_int_distribution<> idx(0, trainSize - 1);
	//data for model
    
    sData<double> x_train = make_shared<Data<double>>(trainSize,input_size);
    sData<double> t_train = make_shared<Data<double>>(trainSize,output_size);

    ReadMNIST(x_train->data,trainSize, "../data/train-images.idx3-ubyte");
    ReadMNISTLabel(t_train->data,trainSize, "../data/train-labels.idx1-ubyte");
  
    sData<double> x_train_batch = make_shared<Data<double>>(batch_size,input_size);     
    sData<double> t_train_batch = make_shared<Data<double>>(batch_size,output_size);

    
	//test for model
	
    sData<double> x_test = make_shared<Data<double>>(testSize,input_size);
    sData<double> t_test = make_shared<Data<double>>(testSize,output_size);
	ReadMNIST(x_test->data,testSize, "../data/t10k-images.idx3-ubyte");
	ReadMNISTLabel(t_test->data,testSize, "../data/t10k-labels.idx1-ubyte");


    int iter_per_epoch = trainSize / batch_size > 1 ? trainSize / batch_size : 1;
    vector<sData<double>> backprop;
    for(int i = 0; i < iteration; i++)
    {
        
        for (int j = 0; j < batch_size; j++)
        {
            int target = idx(mersenne);
            memcpy(&(x_train_batch->data[j*input_size]), &(x_train->data[target*input_size]), sizeof(double) * input_size);
            memcpy(&(t_train_batch->data[j*output_size]), &(t_train->data[target*output_size]), sizeof(double) * output_size);
        }
        
        
        network->gradient(backprop,x_train_batch,t_train_batch);
        
        //network->update(backprop,lRate);
        optimizer->update(network->params,backprop);
        double loss = network->loss(x_train_batch,t_train_batch,false);
        //cout <<"loss : " <<loss <<"\n";
        
        if(i % iter_per_epoch==0)
        {
            double train_acc = network->accuracy(x_train,t_train);
            double test_acc = network->accuracy(x_test,t_test);
            cout <<"\n";
            cout << "train acc : " << train_acc << " test acc : " << test_acc <<"\n";
        }
        
    }                    
	
	delete network;
    delete optimizer;
	return 0;      
}
