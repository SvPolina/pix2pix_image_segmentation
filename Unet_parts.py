import torch
import utils
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Convolutional block that keeps the dim of input
class conv_block(nn.Module):
    def __init__(self,channels_in, channels_out):
        super(conv_block,self).__init__()
        self.conv_b=nn.Sequential(
                      nn.Conv2d(channels_in, channels_out,3,padding=1),
                      nn.BatchNorm2d( channels_out),
                      nn.ReLU(inplace=True)
                  ) 

    def forward (self,x):
          output= self.conv_b(x)
          return output  

# Convolutional block that makes dim_input/2
class conv_block_2(nn.Module):
    def __init__(self,channels_in, channels_out):
        super(conv_block_2,self).__init__()
        self.conv_b=nn.Sequential(
                      nn.Conv2d(channels_in, channels_out,4,stride=2,padding=1),
                      nn.BatchNorm2d( channels_out),
                      nn.ReLU(inplace=True)
                  ) 

    def forward (self,x):
          output= self.conv_b(x)
          return output           

class conv_block_mod(nn.Module):
    def __init__(self,channels_in, channels_out):
        super(conv_block_mod,self).__init__()
        self.conv_b=nn.Sequential(
                      nn.Conv2d(channels_in, channels_out,3,padding=1), 
                      nn.ReLU(inplace=True) ) 

    def forward (self,x):
          output= self.conv_b(x)
          return output           

class contraction_block(nn.Module): 
      def __init__(self,channels_in, channels_out):
        super(contraction_block,self).__init__() 
        self.block=nn.Sequential(conv_block_2(channels_in, channels_out))
      def forward(self,x):
          output=self.block(x)
          return output

class expantion_block(nn.Module):
      def __init__(self, channels_in, channels_out):
          super(expantion_block,self).__init__()
          self.conv_t=nn.ConvTranspose2d(channels_in//2,channels_in//2,2,stride=2)
          self.conv=conv_block(channels_in, channels_out)  

      def stack(self, x_1,x_2):
          dy  = x_2.size()[2] - x_1.size()[2]
          dx  =  x_2.size()[3] - x_1.size()[3]
          x_1 = F.pad(x_1, (dx // 2, dx - dx//2, dy // 2, dy - dy//2))
          output = torch.cat([x_2, x_1], dim=1)
          return output

      def forward(self,x_1,x_2):
          output=self.conv_t(x_1)
          output=self.stack(output,x_2)
          output=self.conv(output)
          return output

class expantion_block_drop(nn.Module):
      def __init__(self, channels_in, channels_out):
          super(expantion_block,self).__init__()
          self.conv_t=nn.ConvTranspose2d(channels_in//2,channels_in//2,2,stride=2)
          self.drop_out=nn.Dropout(p=0.5)
          self.conv=conv_block(channels_in, channels_out)  

      def stack(self, x_1,x_2):
          dy  = x_2.size()[2] - x_1.size()[2]
          dx  =  x_2.size()[3] - x_1.size()[3]
          x_1 = F.pad(x_1, (dx // 2, dx - dx//2, dy // 2, dy - dy//2))
          output = torch.cat([x_2, x_1], dim=1)
          return output

      def forward(self,x_1,x_2):
          output=self.conv_t(x_1)
          output=self.stack(output,x_2)
          #added
          output=self.drop_out(output)
          output=self.conv(output) 
          return output


class conv_block_s(nn.Module):
    def __init__(self,channels_in,channels_out,drop=True,relu=True):
        super(conv_block_s,self).__init__()
        self.block=nn.Sequential(nn.Conv2d(channels_in, channels_out,3,padding=1),
                                 nn.BatchNorm2d( channels_out))
        self.rl=nn.ReLU(inplace=True)
        self.drop_out=nn.Dropout(p=0.5)
        self.drop=drop
        self.relu=relu
        
    def forward(self,x):
        output=self.block(x)
        if self.drop:
           output=self.drop_out(output)
        if self.relu:
           output=self.rl(output)
        return output

class expantion_block_s(nn.Module):
      def __init__(self, channels_in, channels_out,drop=True,relu=True):
          super(expantion_block_s,self).__init__()
          self.conv_t=nn.ConvTranspose2d(channels_in//2,channels_in//2,2,stride=2) 
          nn.init.xavier_normal_(self.conv_t.weight)   
          self.conv=conv_block_s(channels_in, channels_out,drop=True,relu=True)  

      def stack(self, x_1,x_2):
          dy  = x_2.size()[2] - x_1.size()[2]
          dx  =  x_2.size()[3] - x_1.size()[3]
          x_1 = F.pad(x_1, (dx // 2, dx - dx//2, dy // 2, dy - dy//2))
          output = torch.cat([x_2, x_1], dim=1)
          return output

      def forward(self,x_1,x_2):
          output=self.conv_t(x_1)
          output=self.stack(output,x_2)
          output=self.conv(output)  
          return output                       
