import torch
import torch.nn as nn
import torch.nn.functional as F


#Resnet < 50 network 
class ResidualBlock(nn.Module):

	def __init__(self,input_channel,output_channel,stride):
		super(ResidualBlock,self).__init__()
		self.conv1 = nn.Conv2d(input_channel,output_channel,kernel_size=3,stride=stride,padding=1,bias=False)
		self.batch1 = nn.BatchNorm2d(output_channel)
		self.relu1 = nn.ReLU()
		self.conv2 = nn.Conv2d(output_channel,output_channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.batch2 = nn.BatchNorm2d(output_channel)

		#when input data channel  is different from output data channel
		#and input h,w are different from output h,w
		self.shortcut = nn.Sequential()
		if input_channel != output_channel :
			shortcut = nn.Sequential(
					nn.Conv2d(input_channel,output_channel,kernel_size = 1, stride = 2,bias=False),
					nn.BatchNorm2d(output_channel)
					)		
		self.relu2 = nn.Relu()

		def forward(self,input):
			out = self.relu1(self.batch1(self.conv1(input)))
			out = self.batch2(self.conv2(out))
			out += self.shortcut(input)
			out = self.relu2(out)
			return out										


class ResNet(nn.Module):

	def __init__(self,block,input_channels,output_channels):
		super(ResNet,self).__init__()
		self.conv = nn.Conv2d(in_channels = 3, out_channels = 64,kernel_size=3,stride = 1,padding = 1,bias = False)
		self.batch = nn.BatchNorm2d(64)
		self.relu = nn.Relu()
		self.block1 = self.make_block(block[0],1,64,64)
		self.block2 = self.make_block(block[1],2,64,128)
		self.block3 = self.make_block(block[2],2,128,256)
		self.block4 = self.make_block(block[3],2,256,512)
		self.avgPool = nn.AvgPool2d(4)
		self.fc = nn.Linear(512*1,10)

	def forward(self,input):
		out = self.relu(self.batch(self.conv1(input)))
		out = self.block1(out)
		out = self.block2(out)
		out = self.block3(out)
		out = self.block4(out)
		out = self.avgPoll(out)
		out = out.view(out.size(0),-1)
		out = self.fc(out)
		return out
		

	
	def make_block(num_block,initStride,input_channel,output_channel)
		layers = []
		stride = [initStride] + [1]*(num_block-1)
		for s in stride
			layers.append( ResidualBlock(input_channel,output_channel,s))
		return nn.Sequential(*layers)
