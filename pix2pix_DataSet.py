import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os,argparse,time
import scipy.ndimage.morphology as morph
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from torchvision.transforms.functional import to_tensor,to_pil_image
from albumentations import ( HorizontalFlip, VerticalFlip, Compose, Resize,Normalize,Crop)
import copy
import random
import h5py

class px2px_DataSet_h5(data.Dataset):
     def __init__(self,mode,path):
       super(px2px_DataSet_h5, self).__init__()
       self.path=path 
       self.mode=mode
       self.file = h5py.File(self.path+'/cityscapes_im.hdf5', 'r')
       
     def __getitem__(self,idx):
        img = self.file[self.mode][idx, ...]
        crop_im_y=Crop(0,0,256,256)
        crop_im_x=Crop(256,0,512,256)
        im_x=crop_im_x(image=np.array(img))["image"]
        im_y=crop_im_y(image=np.array(img))["image"] 
        p=random.random()
        p_1=random.random()

        if (p>0.5)&(self.mode=="train"):
            vf=VerticalFlip(1)
            im_x=vf(image=np.array(im_x))["image"]
            im_y=vf(image=np.array(im_y))["image"]

        if (p_1>0.5)&(self.mode=="train"):
            hf=HorizontalFlip(1)
            im_x=hf(image=np.array(im_x))["image"]
            im_y=hf(image=np.array(im_y))["image"]

        x=to_tensor(im_x)
        y=to_tensor(im_y)
        return x,y 

     def __len__(self):
        return len(self.path)



class px2px_DataSet(data.Dataset):
     def __init__(self,mode,path):
       self.path=path 
       self.mode=mode
       self.img_paths=os.listdir(self.path+'/data/cityscapes_'+self.mode)
       
     def __getitem__(self,idx):
        img = Image.open(self.path+'/data/cityscapes_'+self.mode + "/" + self.img_paths[idx]).convert('RGB')
        crop_im_y=Crop(0,0,256,256)
        crop_im_x=Crop(256,0,512,256)
        im_x=crop_im_x(image=np.array(img))["image"]
        im_y=crop_im_y(image=np.array(img))["image"] 
        p=random.random()
        p_1=random.random()

        if (p>0.5)&(self.mode=="train"):
            vf=VerticalFlip(1)
            im_x=vf(image=np.array(im_x))["image"]
            im_y=vf(image=np.array(im_y))["image"]

        if (p_1>0.5)&(self.mode=="train"):
            hf=HorizontalFlip(1)
            im_x=hf(image=np.array(im_x))["image"]
            im_y=hf(image=np.array(im_y))["image"]

        x=to_tensor(im_x)
        y=to_tensor(im_y)
        return x,y 

     def __len__(self):
        return len(self.img_paths)