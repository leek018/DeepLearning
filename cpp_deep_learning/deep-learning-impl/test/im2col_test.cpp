#include <iostream>
#include "im2col.h"

using namespace std;


int main()
{
    int N = 1;
    int C = 1;
    int H = 2;
    int W = 2;
    
    int FN = 1;
    int FH = 2;
    int FW = 2;
    
    int stride = 1;
    int pad = 1;
    double *im = new double[N*C*H*W];
    
    int OH = (H+2*pad-FH)/stride+1;
    int OW = (W+2*pad-FW)/stride+1;
    
    double *col = new double[N*FH*FW*C*OH*OW];
    

    for(int i = 0 ; i < N; i++)
    {
        for(int j = 0 ; j < C; j++)
        {
            for(int r = 0; r < H; r++)
            {
                for(int c = 0 ; c < W; c++)
                {
                    im[i*N+C*j+r*H+c] = N*i+C*j+r*H+c;
                }
                cout <<"\n";
            }
        }
    }

    cout <<"input : \n";
    
    for(int i = 0 ; i < N; i++)
    {
        for(int j = 0 ; j < C; j++)
        {
            for(int r = 0; r < H; r++)
            {
                for(int c = 0 ; c < W; c++)
                {
                    cout << im[i*N+C*j+r*H+c] << " ";
                }
                cout <<"\n";
            }
            cout <<"============channel===========\n";
        }
        cout <<"============batch==========\n";
    }
    im2col_cpu<double>(im,C,H,W,FH,stride,pad,col);
    cout <<"output : \n";    
    for(int i = 0 ; i < N; i++)
    {
        for(int j = 0; j < FH*FW*C; j++)
        {
            for(int k = 0; k < OH*OW; k++)
            {
                cout << col[i*FH*FW*C*OH*OW+j*OH*OW+k] <<" ";
            }
            cout <<"\n";
        }
        cout <<"================batch============\n";
    }
    return 0;   
}
