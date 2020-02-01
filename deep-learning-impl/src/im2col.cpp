#include "im2col.h"
template <typename T>
T im2col_get_pixel(T *im, int height, int width,
	int row, int col, int channel_idx, int pad)
{
	row -= pad;
	col -= pad;
	if (row < 0 || col < 0 ||
		row >= height || col >= width) return 0;
	int col_idx = col + width * (row + height * channel_idx);
	return im[col_idx];
}

template <typename T>
void im2col_cpu(T* data_im,
	int channels, int im_height, int im_width,
	int ksize, int stride, int pad, T* data_col)
{
	int output_height = (im_height + 2 * pad - ksize) / stride + 1;
	int output_width = (im_width + 2 * pad - ksize) / stride + 1;

	int channels_col = channels * ksize * ksize;
	for (int c = 0; c < channels_col; ++c) {
		int w_offset = c % ksize;
		int h_offset = (c / ksize) % ksize;
		int channel_idx = c / ksize / ksize;
		for (int h = 0; h < output_height; ++h) {
			for (int w = 0; w < output_width; ++w) {
				int im_row = h_offset + h * stride;
				int im_col = w_offset + w * stride;
				int col_index = (c * output_height + h) * output_width + w;				
				data_col[col_index] = im2col_get_pixel<T>(data_im, im_height, im_width,
					im_row, im_col, channel_idx, pad);
			}			
		}		
	}
}
template void im2col_cpu<double>(double*,int,int,int,int,int,int,double*); 
template void im2col_cpu<float>(float*,int,int,int,int,int,int,float*);
template double im2col_get_pixel<double>(double*,int,int,int,int,int,int);
template float im2col_get_pixel<float>(float*,int,int,int,int,int,int);
