#ifndef _DATA_
#define _DATA_
#include <iostream>
#include <memory>
#include <vector>
#include <cstring>
using namespace std;
template <typename T>
class Data{
public:
    //N,C,H,W
    vector<int> params;
    T *data;
    int size;
    //argNum : The number of argument
    //{N,C,H,W} = {0,1,2,3}
    Data(vector<int> &params_):params(params_)
    {
        int total_size = 1;
        for(int i = 0 ; i < params.size(); i++)
        {
            int paramVal = params[i];
            total_size *= params[i];
        }
        data = new T[total_size];
        size = total_size;
        memset(data,0,sizeof(T)*total_size);
    }
    ~Data()
    {
        if(data!=NULL)
            delete[] data;
        data =NULL;
    }
};
template<typename T>
using sData = shared_ptr<Data<T>>;
#endif

