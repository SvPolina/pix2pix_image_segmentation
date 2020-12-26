import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils import data
import os,argparse,time
import numpy as np
from torchsummary import summary
from torchvision.transforms.functional import to_tensor,to_pil_image



class pix2pix_discriminator(nn.Module):
      def __init__(self,channels=64):
        super(pix2pix_discriminator,self).__init__()
        self.conv_b1 = conv_bn_Lrelu(6,channels) 
        self.conv_b2 = conv_bn_Lrelu(channels,2*channels)
        self.conv_b3 = conv_bn_Lrelu(2*channels,4*channels)
        self.conv_b4 = conv_bn_Lrelu(4*channels,8*channels) 
        self.conv_b5 = conv_bn_Lrelu(8*channels,8*channels,pad=1)
        self.conv_f  = nn.Conv2d(8*channels,1,4,1)

      def forward(self,x):
        out_1=self.conv_b1(x)
        out_2=self.conv_b2(out_1)
        out_3=self.conv_b3(out_2)
        out_4=self.conv_b4(out_3)
        out_5=self.conv_b5(out_4)
        out_6=torch.sigmoid(self.conv_f(out_5))
        return out_6 

class conv_bn_Lrelu(nn.Module):
    def __init__(self,channels_in,channels_out,ker=4,pad=2):
      super(conv_bn_Lrelu,self).__init__()
      self.c_block=nn.Sequential(nn.Conv2d(channels_in,channels_out,kernel_size=ker,padding=pad),
                                 nn.BatchNorm2d(channels_out),
                                 nn.LeakyReLU(0.2)) 
    def forward(self,x):
        output=self.c_block(x)
        return output  

class conv_Lrelu(nn.Module):
    def __init__(self,channels_in,channels_out,ker=4,pad=2):
      super(conv_bn_Lrelu,self).__init__()
      self.c_block=nn.Sequential(nn.Conv2d(channels_in,channels_out,kernel_size=ker,padding=pad),
                                 nn.LeakyReLU(0.2)) 
    def forward(self,x):
        output=self.c_block(x)
        return output       