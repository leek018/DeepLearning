#include <iostream>
#include <vector>
#include <memory>
using namespace std;

template <typename T>
class variable {
public:
	int r, c;
	T* data;
	variable(int r_, int c_) :r(r_), c(c_)
	{
		data = new T[r * c];
	}
	~variable()
	{
		delete[] data;
	}
};
template<typename T>
using sData = shared_ptr<T>;
int main()
{
    vector<sData<double>> v;
    v.push_back(make_shared<Data<double>>(1,3));
    cout << v[0].use_count(0);
    return 0;
}
   
