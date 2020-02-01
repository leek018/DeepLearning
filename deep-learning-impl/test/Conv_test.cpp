#include <iostream>
#include "Convolution.h"
#include <vector>
using namespace std;

int main()
{
    int batch =1;
    int channel = 1;
    int im_width = 2;
    int im_height = 2;
    int FN = 1;
    int FC = 1;
    int FH = 2;
    int FW = 2; 
    int stride = 1;
    int pad = 1;
    vector<int> input_param(4);
    vector<int> kernel_param(4);
    vector<int> bias_param(4);
    input_param[0] = batch;
    input_param[1] = channel;
    input_param[2] = im_width;
    input_param[3] = im_height;
    
    kernel_param[0] = FN;
    kernel_param[1] = FC;
    kernel_param[2] = FH;
    kernel_param[3] = FW;
    
    bias_param[0] = 1;
    bias_param[1] = FN;
    bias_param[2] = 1;
    bias_param[3] = 1;
    
   
    sData<double> input = make_shared<Data<double>>(input_param);
    sData<double> kernel = make_shared<Data<double>>(kernel_param);
    sData<double> bias = make_shared<Data<double>>(bias_param);
    
    //input init
    int total_size = 1;
    for(int i = 0 ; i < input->params.size(); i++)
        total_size *= input->params[i];
    
    for(int c = 0; c < channel; c++)
    {
        for(int i = 0; i < im_width*im_height; i++)
            input->data[c*im_width*im_height+i] = 10*c+i;
    }

    //kernel init
    total_size = 1;
    for(int i = 0 ; i < kernel->params.size(); i++)
        total_size *= kernel->params[i];
    
    fill_n(kernel->data,total_size,1);
    
    
    //bias init
    fill_n(bias->data,FN,0);    





    Convolution *conv = new Convolution(stride,pad,kernel,bias);

    cout << "input : \n";
    for(int c = 0; c < channel; c++)
    {
        for(int r = 0 ; r < im_height; r++)
        {
            for(int w = 0 ; w < im_width; w++)
            {
                cout << input->data[c*im_height*im_width+r*im_width+w] <<" ";
            }
            cout <<"\n";
        }
        cout <<"==================================\n";
    }
    
    cout << "kernel : \n";
    for(int c = 0 ; c < channel; c++)
    {
        for(int r = 0; r < FH; r++)
        {
            for(int w = 0 ; w < FW; w++)
            {
                cout << kernel->data[c*FH*FW + r*FW + w] << " ";
            }
            cout <<"\n";
        }
        cout <<"================\n";
    }
    
    cout <<"bias : \n";
    for(int i = 0 ; i < FN; i++)
    {
        cout << bias->data[i] <<" ";
    }
    cout <<"\n";
    
    sData<double> output = conv->forward(input);
    
    int OH = output->params[2];
    int OW = output->params[3];

    cout <<"========forward==============\n";
    cout <<"output : \n";
    for(int i = 0 ; i < FN; i++)
    {
        for(int j = 0 ; j < OH; j++)
        {
            for(int k = 0 ; k < OW; k++)
            {
                cout << output->data[i*OH*OW+j*OW+k];
            }
            cout <<"\n";
        }
        cout <<"===============\n";
    }
    sData<double> &col = conv->col; 
    cout << "col : \n";
    for(int n = 0; n < batch; n++)
    {
        for(int r = 0; r < FH*FW*channel; r++)
        {
            for(int w = 0; w < OH*OW; w++)
            {
                cout << col->data[n*FH*FW*channel*OH*OW+r*OH*OW+w] <<" ";
            }
            cout <<"\n";
        }
        cout <<"================\n";
    }
    
    // dout init
    vector<int> dout_param(4);
    dout_param[0] = batch;
    dout_param[1] = FN;
    dout_param[2] = OH;
    dout_param[3] = OW;

    sData<double> dout = make_shared<Data<double>>(dout_param);
    total_size = 1;
    for(int i = 0 ; i < dout->params.size(); i++)
        total_size *= dout->params[i];
    
    for(int i = 0 ; i < total_size; i++)
    {
        dout->data[i] = 1;
    }
    
    cout <<"test dout: \n";
    for(int i = 0; i < batch; i++)
    {
        for(int j = 0 ; j < FN*OH*OW; j++)
        {
            cout << dout->data[i*FN*OH*OW+j] <<" ";
        }
        cout <<"\n";
    }
     
    cout <<"========backward==============\n";
    sData<double> dX = conv->backward(dout);

    cout <<"dX : \n";
    for(int c = 0; c < channel; c++)
    {
        for(int r = 0 ; r < im_height; r++)
        {
            for(int w = 0 ; w < im_width; w++)
            {
                cout << dX->data[c*im_height*im_width+r*im_width+w] <<" ";
            }
            cout <<"\n";
        }
        cout <<"==================================\n";
    }
    sData<double> &dW = conv->dK;
    cout << "dW : \n";
    for(int c = 0 ; c < channel; c++)
    {
        for(int r = 0; r < FH; r++)
        {
            for(int w = 0 ; w < FW; w++)
            {
                cout << dW->data[c*FH*FW + r*FW + w] << " ";
            }
            cout <<"\n";
        }
        cout <<"================\n";
    }
    
    sData<double> &dB = conv->dB;
    cout <<"dB : \n";
    for(int i = 0 ; i < FN; i++)
    {
        cout << dB->data[i] <<" ";
    }
    cout <<"\n";
    return 0;
}

    
