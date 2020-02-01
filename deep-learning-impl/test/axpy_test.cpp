#include <iostream>
extern "C"{
    #include <cblas.h>
}
using namespace std;

int main()
{
    int row = 3;
    int col = 3;
    double *A = new double[row*col];
    for(int i = 0 ; i < row*col; i++)
        A[i] = i;
    double *B = new double[row*col];
    double alpha = 5;
    double *A_copy = A;
    double *B_copy = B;
    for(int i = 0; i < col; i++){        
        cblas_daxpy(row,alpha,A_copy,col,B_copy,col); 
        A_copy++;
        B_copy++;
    }
    cout <<"A : \n";
    for(int i = 0 ; i < row; i++)
    {
        for(int j = 0 ; j < col; j++)
            cout << A[i*col+j] << " ";
        cout <<"\n";
    }

    cout << "alpha : " << alpha <<"\n";
    
    cout <<"B : \n";
    for(int i = 0 ; i < row; i++)
    {
        for(int j = 0 ; j < col; j++)
            cout << B[i*col+j] << " ";
        cout <<"\n";
    }
    return 0;


}
