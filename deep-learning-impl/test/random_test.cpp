#include <iostream>
#include <random>

using namespace std;

int main()
{
    random_device rd;
	mt19937 mersenne(rd());
    int trainSize = 100;
	uniform_int_distribution<> idx(0, trainSize - 1);
    int howmany;
    cin >> howmany;
    for(int time = 0 ; time < howmany; time++)
    {
        for(int i = 0 ; i < 5; i++){
            int target = idx(mersenne);
            cout << target <<"\n";
        }
        cout <<"==============\n";
    }    
   
}
