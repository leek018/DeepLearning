# Naive C++ implementation of Deep learning layers

+ Data
  + Tensor
    + sData<T> : shared_ptr<Data<T>>
      + Data<T>(vector<int> params)
        + params
          + Four dimention
          + params[0] : N
          + params[1] : C
          + parmas[2] : H
          + params[3] : W
        + data
          + flatten array
+ Layer
  + Affine(fully connected layer)
    + Y = AW+B
  + Convolution
    + im2col
    + col2im
  + Relu
  + Pooling
    + Maxpool
  + SoftMaxWithLoss
  + Sigmoid
  + BatchNormalization
    + CNN
    + Dense
  + Optimizer
    + Adam

## Require

+ Blas Libaray
  + cblas
    + Atlas

##  History

### 20-02-01

+ Batchnorm,Affine,Relu,Sigmoid,SoftMaxWithLoss,Adam,Convolution,Maxpooling -CPU
