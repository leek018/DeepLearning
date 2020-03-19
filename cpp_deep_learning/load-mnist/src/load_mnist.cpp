#include <iostream>
#include <fstream>
#include <cstring>
#include "load_mnist.h"
using namespace std;

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
void ReadMNIST(double *Data,int NumberOfImages,string file_name)
{	
	ifstream file(file_name, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		
		cout << "MNIST DATA SET\n";
		cout << "rows : " << n_rows << "\n";
		cout << "cols : " << n_cols << "\n";
		cout << "normalize by 255\n";		
		for (int i = 0; i < NumberOfImages*n_rows*n_cols; ++i)
		{
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            Data[i] = (double)temp/255;
		}
	}
    else{
        cout <<"Train Data Not Found\n";
    }
}

void ReadMNISTLabel(double *label,int NumberOfImage,string file_name) {
	// 레이블을 읽어온다.
    memset(label,0,sizeof(double)*NumberOfImage*10);
	ifstream file(file_name);
	if (file.is_open())
	{
		for (int i = 0; i < 8; i++)
		{
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
		}
		for (int i = 0; i < NumberOfImage; ++i)
		{
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			int val = (int)temp;
			if (i == 1)
				cout << "read label test : " << val << "\n";
			label[val+10*i] = 1;
		}
	}
    else{
        cout <<"Label Not Found\n";
    }
}
