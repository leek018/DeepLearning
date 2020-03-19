#include <iostream>
#include <cstring>
#include <cmath>
#include "BatchNormalization.h"
extern "C"{
    #include <cblas.h>
}
sData<double> BatchNormalization::forward(sData<double> x,bool train_flag)
{
    int N = x->params[0];
    int C = x->params[1];
    int H = x->params[2];
    int W = x->params[3];
    vector<int> running_params(4);
    sData<double> output = make_shared<Data<double>>(x->params);
    int running_len = W;
    if(C == 1 && H == 1){           
        running_params[0] = 1;
        running_params[1] = 1;
        running_params[2] = 1;
        running_params[3] = W;
    }
    else 
    {
        running_params[0] = 1;
        running_params[1] = C;
        running_params[2] = 1;    
        running_params[3] = 1;
        running_len = C;
    }
    if(running_mean->size == 0)
    {
        running_mean = make_shared<Data<double>>(running_params);
        running_var = make_shared<Data<double>>(running_params);
    }
    sData<double> output_xc,output_xhat;
    //When before layer was Affine
    if(train_flag)
    {
        xhat = make_shared<Data<double>>(x->params);
        xc = make_shared<Data<double>>(x->params);    
        memcpy(xc->data,x->data,sizeof(double)*(x->size));
        std = make_shared<Data<double>>(running_params);
        double *mu = new double[running_len];
        memset(mu,0,sizeof(double)*running_len);
        double *xc_square = new double[x->size];
        double *var = new double[running_len];
        memset(var,0,sizeof(double)*running_len);

        if( C == 1 && H == 1)
        {
            //mean process
            int m = N; 
            int n = W;

            //vector for Transposed Matrix
            double *sum_vector = new double[N];
            
            fill_n(sum_vector,m,(double)1/N);         
            cblas_dgemv(CblasRowMajor,CblasTrans,m,n,1,x->data,n,sum_vector,1,0.0,mu,1);
            
            //xc process
            for(int j = 0 ; j < W; j++)
            {                
                int muVal = mu[j];
                for(int i = 0 ; i < N; i++)
                    xc->data[i*W+j] -= muVal;
            } 
            
            //var process
            for(int i = 0 ; i < N*W; i++)
                xc_square[i] = xc->data[i]*xc->data[i];
            cblas_dgemv(CblasRowMajor,CblasTrans,m,n,1,xc_square,n,sum_vector,1,0.0,var,1);
            
            //std process
            for(int i = 0 ; i < W; i++)
                std->data[i] = sqrt(var[i]);
           
            //xhat process
            for(int j = 0 ; j< W; j++)
            {
                double stdVal = std->data[j] + 10e-7;
                for(int i = 0 ; i < N; i++)
                    xhat->data[i*W+j] = xc->data[i*W+j] / stdVal;
            }
            output_xc = xc;
            output_xhat = xhat;
            delete[] sum_vector;
        }
        else
        {
            //again mu
            double scale = 1./(N*W*H);
            for(int i = 0 ; i < C; i++)
            {
                for(int j = 0; j < N; j++)
                {
                    for(int k = 0; k < W*H; k++)
                        mu[i]+= x->data[(i+j*C)*W*H+k];
                }
                mu[i] *= scale;
            }
            
            //xc
            for(int i = 0 ; i < C; i++)
            {
                double muVal = mu[i];
                for(int j = 0; j < N; j++)
                {
                    for(int k = 0; k < W*H; k++)
                        xc->data[(i+j*C)*W*H+k] -= muVal;
                }
            }        
            
            //var
            for(int i = 0 ; i < N*C*H*W; i++)
                xc_square[i] = xc->data[i]*xc->data[i];
            for(int i = 0 ; i < C; i++)
            {
                for(int j = 0; j < N; j++)
                {
                    for(int k = 0; k < W*H; k++)
                        var[i]+= xc_square[(i+j*C)*W*H+k];
                }
                var[i] *= scale;
            }
            
            //std
            for(int i = 0 ; i < C; i++)
                std->data[i] = sqrt(var[i]);
            
            //normalize
            for(int i = 0 ; i < C; i++)
            {
                double stdVal = std->data[i] + 10e-7;
                for(int j = 0; j < N; j++)
                {
                    for(int k = 0; k < W*H; k++)
                        xhat->data[(i+j*C)*W*H+k] = xc->data[(i+j*C)*W*H+k] / stdVal;
                }
            }
        }
        for(int i = 0 ; i < running_len; i++)
        {
            running_mean->data[i] = momentum*running_mean->data[i] + (1-momentum)*mu[i];                
            running_var->data[i] = momentum*running_var->data[i] + (1-momentum)*var[i];
        }
        delete[] mu;
        delete[] xc_square;
        delete[] var;

    }
    else
    {
        output_xc = make_shared<Data<double>>(x->params);        
        output_xhat = make_shared<Data<double>>(x->params);
        memcpy(output_xc->data,x->data,sizeof(double)*(x->size));
        if( C == 1 && H == 1)
        {
            for(int j = 0 ; j < W; j++)
            {
                double rmeanVal = running_mean->data[j];
                double rVarVal = sqrt(running_var->data[j]) + 10e-7;
                for(int i = 0 ; i < N; i++)
                {
                    output_xc->data[i*W+j] -= rmeanVal;
                    output_xhat->data[i*W+j] = xc->data[i*W+j] / rVarVal;
                }
            }            
        }
        else
        {
            for(int i = 0 ; i < C; i++)
            {
                double rmeanVal = running_mean->data[i];
                double rVarVal = sqrt(running_var->data[i]) + 10e-7;
                for(int j = 0; j < N; j++)
                {
                    for(int k = 0; k < W*H; k++){
                        output_xc->data[(i+j*C)*W*H+k ] -= rmeanVal;
                        output_xhat->data[(i+j*C)*W*H+k ] = xc->data[(i+j*C)*W*H+k] / rVarVal;
                    }
                }
            }
        }        
    }
    //scale and shift
    if( C == 1 && H == 1)
    {
        for(int j = 0 ; j < W; j++)
        {
            double gammaVal = gamma->data[j];
            double betaVal = beta->data[j];
            for(int i = 0 ; i < N; i++)
                output->data[i*W+j] = gammaVal*output_xhat->data[i*W+j] + betaVal;
        }        
    }
    else{

        for(int i = 0 ; i < C; i++)
        {
            double gammaVal = gamma->data[i];
            double betaVal = beta->data[i];
            for(int j = 0; j < N; j++)
            {
                for(int k = 0; k < W*H; k++){
                    output->data[ (i+j*C)*W*H+k] = gammaVal*output_xhat->data[ (i+j*C)*W*H+k  ]+betaVal; 
                }
            }
        }
    }
    return output;        
}

sData<double> BatchNormalization::backward(sData<double> dout)
{
    int N = dout->params[0];
    int C = dout->params[1];
    int H = dout->params[2];
    int W = dout->params[3];
    
    vector<int> backward_params = beta->params;
    dbeta = make_shared<Data<double>>(backward_params);
    dgamma = make_shared<Data<double>>(backward_params);
   
    sData<double> dxhat = make_shared<Data<double>>(dout->params);
    sData<double> dvar = make_shared<Data<double>>(backward_params);
    sData<double> dxc = make_shared<Data<double>>(dout->params);
    
    sData<double> dmu = make_shared<Data<double>>(backward_params);
    sData<double> dx = make_shared<Data<double>>(dout->params);
    double scaleFactor = 2./N;
    if( C == 1 && H == 1)
    {
        //dbeta process
        double *sum_vector = new double[W];
        fill_n(sum_vector,W,1);        
        cblas_dgemv(CblasRowMajor,CblasTrans,N,W,1,dout->data,W,sum_vector,1,0.0,dbeta->data,1);

        
        /*** dbeta process debug ***/
        cout << "beta process debug : \n";
        for(int j = 0 ; j < W; j++)
        {
            cout << dbeta->data[j] <<" ";
        }
        cout <<"\n";
        
        //dgamma process
        for(int j = 0 ; j < W; j++)
        {
            double sum = 0;
            for(int i = 0 ; i < N; i++)
                sum += xhat->data[i*W+j]*dout->data[i*W+j];
            dgamma->data[j] = sum;
        }

        /*** dgamma process debug ***/
        cout << "xhat : \n";
        for(int i = 0 ; i < N; i++)
        {
            for(int j = 0 ; j < W; j++)
            {
                cout << xhat->data[i*W+j] <<" ";
            }
            cout <<"\n";
        }

        cout << "dgamma process debug : \n";
        for(int j = 0 ; j < W; j++)
        {
            cout << dgamma->data[j] <<" ";
        }
        cout <<"\n";
        //dxhat process
        double *dxhatData = dxhat->data;
        double *doutData = dout->data;
        for(int j = 0 ; j < W; j++)
        {
            double gammaVal = gamma->data[j];
            cblas_daxpy(N,gammaVal,doutData,W,dxhatData,W); 
            dxhatData++;
            doutData++;
        } 

        /*** dxhat process debug ***/
        cout << "dxhat process debug : \n";
        for(int i = 0 ; i < N; i++)
        {
            for(int j = 0 ; j < W; j++)
            {
                cout << dxhat->data[i*W+j] <<" ";
            }
            cout <<"\n";
        }

        //dxc process
        double *dxcData = dxc->data;
        dxhatData = dxhat->data;
        for(int j = 0 ; j < W; j++)
        {
            double stdVal = std->data[j] + 10e-7;           
            cblas_daxpy(N,1./stdVal,dxhatData,W,dxcData,W);
            dxcData++;
            dxhatData++; 
        }
        /*** dxc process debug ***/
        cout << "dxc process debug : \n";
        for(int i = 0 ; i < N; i++)
        {
            for(int j = 0 ; j < W; j++)
            {
                cout << dxc->data[i*W+j] <<" ";
            }
            cout <<"\n";
        }
         
        //dvar process
        for(int j = 0 ; j < W; j++)
        {
            double sum = 0;
            double stdVal_square = std->data[j]*std->data[j] + 10e-7;
            for(int i = 0 ; i < N; i++)
                sum += dxhat->data[i*W+j]*xc->data[i*W+j];
            sum /= stdVal_square;
            sum *= -1./(2*std->data[j]);
            dvar->data[j] = sum;
        }
            
        /*** dvar process debug ***/
        cout << "dvar process debug : \n";
        for(int i = 0 ; i < N; i++)
        {
            for(int j = 0 ; j < W; j++)
            {
                cout << dvar->data[i*W+j] <<" ";
            }
            cout <<"\n";
        }        



        //dmu process
        double *xcData = xc->data;
        double *dxcTemp = new double[dout->size];
        double *dxcTempCopy = dxcTemp;
        for(int j = 0 ; j < W; j++)
        {
            double dvarVal = dvar->data[j];
            cblas_daxpy(N,scaleFactor*dvarVal,xcData,W,dxcTempCopy,W);
            xcData++;
            dxcTempCopy++; 
        }
        for(int i = 0; i < dout->size; i++)
            dxc->data[i] += dxcTemp[i];
        cblas_dgemv(CblasRowMajor,CblasTrans,N,W,1,dxc->data,W,sum_vector,1,0.0,dmu->data,1);
        /*** dxc before dmu process debug ***/
        cout << "dxc before dmu process debug : \n";
        for(int i = 0 ; i < N; i++)
        {
            for(int j = 0 ; j < W; j++)
            {
                cout << dxc->data[i*W+j] <<" ";
            }
            cout <<"\n";
        }
        /*** dmu process debug ***/
        cout << "dmu process debug : \n";
        for(int i = 0 ; i < N; i++)
        {
            for(int j = 0 ; j < W; j++)
            {
                cout << dmu->data[i*W+j] <<" ";
            }
            cout <<"\n";
        }
    
        //dx process
        for(int j = 0 ; j< W; j++)
        {
            double dmuVal = dmu->data[j] / N;
            for(int i = 0 ; i < N; i++)
                dx->data[i*W+j] = dxc->data[i*W+j]-dmuVal;
        }

        /*** dx process debug ***/
        cout << "dx process debug : \n";
        for(int i = 0 ; i < N; i++)
        {
            for(int j = 0 ; j < W; j++)
            {
                cout << dx->data[i*W+j] <<" ";
            }
            cout <<"\n";
        }

        delete[] dxcTemp;
        delete[] sum_vector;   
    }
    else
    {
        for(int i = 0 ; i < C; i++)
        {
            double sum = 0;
            for(int j = 0; j < N; j++)
            {
                for(int k = 0; k < W*H; k++)
                    sum += dout->data[(i+j*C)*W*H+k];
            }
            dbeta->data[i] = sum;
        }

        for(int i = 0 ; i < C; i++)
        {
            double sum = 0;
            for(int j = 0; j < N; j++)
            {
                for(int k = 0; k < W*H; k++)
                    sum += xhat->data[(i+j*C)*W*H+k] * dout->data[(i+j*C)*W*H+k];
            }
            dgamma->data[i] = sum;
        }
      
        //dxhat process
        for(int i = 0 ; i < C; i++)
        {
            double gammaVal = gamma->data[i];
            for(int j = 0; j < N; j++)
            {
                for(int k = 0; k < W*H; k++)
                    dxhat->data[(i+j*C)*W*H+k]= gammaVal * dout->data[(i+j*C)*W*H+k];
            }
        }
        
        //dxc process
        for(int i = 0 ; i < C; i++)
        {
            double stdVal = std->data[i]+10e-7;
            for(int j = 0; j < N; j++)
            {
                for(int k = 0; k < W*H; k++)
                    dxc->data[(i+j*C)*W*H+k]= dxhat->data[(i+j*C)*W*H+k] / stdVal;
            }
        }

        
        //dvar process
        for(int i = 0 ; i < C; i++)
        {
            double sum = 0;
            double stdVal_square = std->data[i]*std->data[i] + 10e-7;
            for(int j = 0; j < N; j++)
            {
                for(int k = 0; k < W*H; k++)
                    sum += dxhat->data[(i+j*C)*W*H+k] * xc->data[(i+j*C)*W*H+k];
            }
            sum /= stdVal_square;
            sum *= -1./(2*std->data[i]);
            dvar->data[i] = sum;
        }
          
        
        //dmu process
        for(int i = 0 ; i < C; i++)
        {            
            double dvarVal = dvar->data[i];
            double sum = 0;
            for(int j = 0; j < N; j++)
            {
                for(int k = 0; k < W*H; k++)
                {
                    dxc->data[(i+j*C)*W*H+k] += scaleFactor * xc->data[(i+j*C)*W*H+k] * dvarVal;
                    sum += dxc->data[(i+j*C)*W*H+k];
                }
            }
            dmu->data[i] = sum;
        }
        
        //dx process
        for(int i = 0 ; i < C; i++)
        {
            double dmuVal = dmu->data[i]/N;
            for(int j = 0; j < N; j++)
            {
                for(int k = 0; k < W*H; k++)
                    dx->data[(i+j*C)*W*H+k]= dxc->data[(i+j*C)*W*H+k] - dmuVal;
            }
        }
    }
    return dx;
} 
