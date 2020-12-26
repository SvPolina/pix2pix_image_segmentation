import random
from pix2pix_DataSet import px2px_DataSet
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from  Unet import Unet
import torchvision
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import os,argparse,time
import scipy.ndimage.morphology as morph
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from torchvision.transforms.functional import to_tensor,to_pil_image
from albumentations import ( HorizontalFlip, VerticalFlip, Compose, Resize,Normalize,Crop)
from pix2pix_discriminator  import pix2pix_discriminator
import copy

class test_pix2pix(object):
  def __init__(self,args,saved_weights=False):
    self.args=args
    self.validation_dataset=px2px_DataSet_h5('val',self.args.path)
    self.G = Unet(3,3)
    if saved_weights:
       self.G.load_state_dict(torch.load(self.args.save_path+'/pix2pix_results'+'/generator.pt'))
    self.G.cuda()

  def run(self):
      train_loader = DataLoader(dataset=self.validation_dataset, batch_size=self.args.batch_size, num_workers=2,shuffle=True) 
      self.samp_x,self.samp_y=train_loader.__iter__().__next__()     
      self.plot_results()   

  def plot_results(self):  
      title=[" ".join(str(k)+"="+str(v)+",") for k,v in sorted(vars(self.args).items())]       
      fig,axs = plt.subplots(self.args.batch_size,3,figsize=(10,10), sharex=True, sharey=True)
      for i in range(self.args.batch_size):
          x=self.samp_x[i,:,:,:,]
          y=self.samp_y[i,:,:,:,] 
          g_y=self.G(x.unsqueeze(0).cuda())
          axs[i, 0].imshow(to_pil_image(x))
          axs[i, 1].imshow(to_pil_image(y))
          axs[i, 2].imshow(to_pil_image(g_y.squeeze().cpu()))
      plt.show()  
      fig.savefig(self.args.save_path+'/pix2pix_results'+'/_test_'+'.png', bbox_inches='tight' )
      fig.clf()
 
def parse_arguments(): 
     parser = HyperOptArgumentParser(strategy='grid_search')
     parser.add_argument('-f')
     parser.add_argument('--num_epochs', type=int, default=1, help='The number of epochs to run') 
     parser.add_argument('--batch_size', type=int, default=3) 
     parser.add_argument('--save_path', type=str, default='/default/pix2pix_results')
     parser.add_argument('--path', type=str, default='/default/path') 
     return parser.parse_args()

def main(saved_w):
    args=parse_arguments() 
    pix2pix=test_pix2pix(args,saved_w)  
    my_model=pix2pix.run() 
    return my_model     

if __name__=='__main__':
    my_model=main(True)
