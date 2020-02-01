#ifndef _COL2IM_
#define _COL2IM_
template <typename T>
void col2im_add_pixel(T *im, int height, int width, int channels,
                        int row, int col, int channel, int pad, T val);
template <typename T>
void col2im_cpu(T* data_col,
         int channels,  int height,  int width,
         int ksize,  int stride, int pad, T* data_im);
#endif
