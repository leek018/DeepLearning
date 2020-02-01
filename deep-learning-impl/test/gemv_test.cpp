#include <iostream>
extern "C"{
    #include <cblas.h>
}
using namespace std;

int main()
{
    /*    
    int ROW = 3;
    int COL = 4;
    double *mat = new double[ROW*COL];
    for(int i = 0 ; i < ROW*COL; i++)
        mat[i] = i; 
    cout << "input : \n";
    for(int i = 0 ; i < ROW; i++)
    {
        for(int j = 0 ; j < COL; j++)
            cout << mat[i*COL+j] << " ";
        cout <<"\n";
    }    
    
    //if Trans => vec[ROW] 
    //ELSE => vec[COL]
    double *vec = new double[ROW];
    for(int i = 0 ; i < ROW; i++)
        vec[i] = i;
    
    cout << "vec : \n";
    for(int i = 0 ; i < ROW; i++)
        cout << vec[i] << "\n";

    //if Trans => OUT[COL] 
    //ELSE => OUT[ROW]    
    double *out = new double[COL];
        
void cblas_dgemv(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 const double *X, const int incX, const double beta,
                 double *Y, const int incY);
enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102 };
   enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113,
                         AtlasConj=114};
    //cblas_dgemv(CblasRowMajor,CblasNoTrans,3,3,1,mat,3,vec,1,0.0,out,1);
    
    //if Trans => M = ROW, N = COL
    //ELSE => M = COL , N = ROW    
    int M = ROW;
    int N = COL;

    //M = COL;
    //N = ROW;

    cblas_dgemv(CblasRowMajor,CblasTrans,M,N,1,mat,N,vec,1,0.0,out,1);
    cout <<"output : \n";
    for(int i = 0 ; i < COL; i++)
        cout << out[i] << "\n";
    return 0;
    */

}    
