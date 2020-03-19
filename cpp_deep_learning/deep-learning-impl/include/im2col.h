#ifndef _IM2COL_
#define _IM2COL_
template <typename T>
T im2col_get_pixel(T *im, int height, int width,int row, int col, int channel_idx, int pad);
template <typename T>
void im2col_cpu(T* data_im,	int channels, int im_height, int im_width,	int ksize, int stride, int pad, T* data_col);
#endif
