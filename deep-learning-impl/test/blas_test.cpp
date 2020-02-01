#include <iostream>
#include <cstring>
#include <vector>
extern "C"{
    #include <cblas.h>
}

using namespace std;

int main()
{
    //vector<double> A;
    //vector<double> B(6);
    double A[2*4] = {1,2,3,4,5,6,7,8};
    double B[2*3] = {-1,-1,-1,-1,-1,-1};
    
    int m = 4;
    int n = 3;
    int k = 2;
    double C[4*3] = {0,};
    //vector<double> C(12);
    /*
    for(int i = 0 ; i < 8; i++)
        A.push_back(i+1);
    for(int i = 0 ; i <6;i++)
        B[i] = -1;
    for(int i = 0 ; i < 8; i++)
        cout << A[i] <<" ";    
    cout <<"\n";
    
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m,n,k,1.0,
                &A[0],m,&B[0],n,0.0,&C[0],n);
    */
    for(int a = 0 ; a < 2; a++)
    {        
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.0,
                &A[0],k,&B[0],n,0.0,&C[0],n);
        for(int i = 0 ; i < m; i++)
        {
            for(int j = 0 ; j <n; j++)
                cout << C[i*n+j] << " ";
            cout <<"\n";
        }
        cout <<"===============\n";
    }


    return 0;
} 
