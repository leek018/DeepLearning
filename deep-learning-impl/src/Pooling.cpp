#include "Pooling.h"
#include <limits>
using namespace std;
sData<double> Pooling::forward(sData<double> x)
{
    int N = x->params[0];
    int C = x->params[1];
    int H = x->params[2]; 
    int W = x->params[3];
    
    int OH = (H+2*pad-pool_h)/stride + 1;
    int OW = (W+2*pad-pool_w)/stride + 1;

    double *input = x->data;
    vector<int> out_params(4);
    out_params[0] = N;
    out_params[1] = C;
    out_params[2] = OH;
    out_params[3] = OW;
    sData<double> output = make_shared<Data<double>>(out_params);
    max_indexes = make_shared<Data<int>>(out_params);
    
    for (int batch = 0; batch < N; batch++)
	{
		for (int channel = 0; channel < C; channel++)
		{
			for (int out_r = 0; out_r < OH; out_r++)
			{
				for (int out_c = 0; out_c < OW; out_c++)
				{
					int input_start_c = out_c * stride - pad;
					int input_start_r = out_r * stride - pad;
					double local_max = -numeric_limits<double>::max();
					int max_i = -1;
					for (int pool_r = 0; pool_r < pool_h; pool_r++)
					{
						for (int pool_c = 0; pool_c < pool_w; pool_c++)
						{
							int input_x = input_start_c + pool_c;
							int input_y = input_start_r + pool_r;
							if (input_x >= 0 && input_x < W && input_y >= 0 && input_y < H)
							{
								int index = batch*H*W*C+channel * H * W + input_y * W + input_x;
								double val = input[index];
								if (local_max < val) {
									local_max = val;
									max_i = index;
								}																	
							}
						}
					}
					int output_index = batch*C*OW*OH+channel * OW * OH + out_r * OW + out_c;
					output->data[output_index] = local_max;
					max_indexes->data[output_index] = max_i;
				}
			}
		}
	} 
    X = x;
    return output;
}    

sData<double> Pooling::backward(sData<double> dout)
{
    sData<double> dx = make_shared<Data<double>>(X->params);
   
    for(int i = 0 ; i < dout->size; i++)
    {
        int index = max_indexes->data[i];
        dx->data[index] = dout->data[i];
    }
    return dx;
}
    


















   
