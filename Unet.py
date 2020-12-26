import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import numpy as np
from  Unet_parts import *


class Unet(nn.Module):
       def __init__(self,in_channels,out_classes):
            super(Unet, self).__init__() 
            #self.start=conv_block(in_channels,64) 
            self.start=conv_block_mod(in_channels,64)
            self.cont_1=contraction_block(64,128)  
            self.cont_2=contraction_block(128,256)  
            self.cont_3=contraction_block(256,512)  
            self.cont_4=contraction_block(512,512) 
            self.exp_1=expantion_block_s(1024,256)#expantion_block(1024,256,True,True) 
            self.exp_2=expantion_block_s(512,128)#expantion_block(1024,256,True,True)
            self.exp_3=expantion_block_s(256,64)
            self.exp_4=expantion_block_s(128,64) 
            self.conv=nn.Conv2d(64,out_classes, 1) 
      
       def forward(self,x):
            output_1=self.start(x)   
            output_2=self.cont_1(output_1) 
            output_3=self.cont_2(output_2)
            output_4=self.cont_3(output_3)
            output_5=self.cont_4(output_4)
            output_6=self.exp_1(output_5,output_4)
            output_7=self.exp_2(output_6,output_3)
            output_8=self.exp_3(output_7,output_2)
            output_9=self.exp_4(output_8,output_1)
            output_10=self.conv(output_9)
            return torch.tanh(output_10)