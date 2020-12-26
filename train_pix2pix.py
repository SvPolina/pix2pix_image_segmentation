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


writer = SummaryWriter('/path/runs')
dat_set=px2px_DataSet_h5('train','/path/cityscapes')
train_loader = DataLoader(dataset=dat_set, batch_size=1, num_workers=4,shuffle=True)


class train_pix2pix(object):
    def __init__(self,args,saved_weights=False):
      self.args=args
      self.train_dataset=px2px_DataSet('train',self.args.path)
      self.G = Unet(3,3)
      self.D = pix2pix_discriminator(64) 

      if saved_weights:
        self.G.load_state_dict(torch.load(self.args.save_path+'/generator.pt'))
        self.D.load_state_dict(torch.load(self.args.save_path+'/discriminator.pt'))
      else:
        utils.init_weights(self.G,self.args.mean,self.args.std)
        utils.init_weights(self.D,self.args.mean,self.args.std) 
        ####  

      self.D.cuda()
      self.G.cuda()
      self.train_hist= {'D_loss': [], 'G_loss': [],'total_time':[] }
      self.BCE_loss=nn.BCELoss().cuda()
      self.L1_loss=nn.L1Loss().cuda()

    def run(self):
        optimizer_G=optim.Adam(self.G.parameters(),lr=self.args.lrG,betas=(self.args.beta1,self.args.beta2))
        optimizer_D=optim.Adam(self.D.parameters(),lr=self.args.lrD,betas=(self.args.beta1,self.args.beta2))
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, num_workers=2,shuffle=True) 
        self.samp_x,self.samp_y=train_loader.__iter__().__next__()   
        start_time=time.time()
        for epoch in range(self.args.num_epochs):
            print("---Epoch #---",epoch)
            self.train_model(optimizer_G,optimizer_D,train_loader,epoch) 
        writer.close()          
        print('---End training---')    
        self.train_hist['total_time'].append((time.time()-start_time))   
        self.plot_loss()
        self.plot_results()    
        torch.save(self.G.state_dict(), self.args.save_path+'/generator.pt')
        torch.save(self.D.state_dict(), self.args.save_path+'/discriminator.pt')

    def train_model(self,optimizer_g,optimizer_d,t_loader,epoch):    
        self.D.train() 
        self.G.train()      

        for iter,cur_batch in enumerate(t_loader):           
            x_,y_=cur_batch
            x_,y_=x_.cuda(),y_.cuda() 
            optimizer_d.zero_grad()
            real_scores_D=self.D(torch.cat([x_,y_],dim=1))
            real_loss_D=self.BCE_loss(real_scores_D,torch.ones(real_scores_D.size()).cuda())
            fake_im=self.G(x_)
            fake_scores_D=self.D(torch.cat([x_,fake_im],dim=1))          
            fake_loss_D=self.BCE_loss(fake_scores_D,torch.zeros(fake_scores_D.size()).cuda())
            total_loss_D=(real_loss_D + fake_loss_D) * 0.5
            self.train_hist['D_loss'].append(total_loss_D.item())
            total_loss_D.backward()
            optimizer_d.step()

            #Train G
            optimizer_g.zero_grad()
            fake_im=self.G(x_)
            fake_scores_D=self.D(torch.cat([x_,fake_im],dim=1))
            l1_loss_G=self.args.lambd*self.L1_loss(fake_im,y_)
            bce_loss_G=self.BCE_loss(fake_scores_D,torch.ones(fake_scores_D.size()).cuda())
            total_loss_G=bce_loss_G+l1_loss_G
            self.train_hist['G_loss'].append(total_loss_G.item())
            total_loss_G.backward()
            optimizer_g.step()

            if iter%5==0: 
              torch.save(self.G.state_dict(), self.args.save_path+'/generator.pt')
              torch.save(self.D.state_dict(), self.args.save_path+'/discriminator.pt')

        writer.add_scalar("Loss_G/train", total_loss_G, epoch)
        writer.add_scalar("Loss_D/train", total_loss_D, epoch)
        writer.flush() 

    def plot_loss(self):
        plt.figure()
        D_loss=plt.plot(self.train_hist['D_loss'])
        G_loss=plt.plot(self.train_hist['G_loss'])      
        plt.legend([D_loss, G_loss], ['Discriminator', 'Generator'], loc='upper right')
        plt.title('Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig(self.args.save_path+'/loss.png', bbox_inches='tight')

    def plot_results(self):  
        title=[" ".join(str(k)+"="+str(v)+",") for k,v in sorted(vars(self.args).items())]       
        fig,axs = plt.subplots(self.args.batch_size,3,figsize=(10,10), sharex=True, sharey=True)
        title="LrG: "+str(self.args.lrG)+"LrD: "+str(self.args.lrD)
        fig.suptitle(title)
        for i in range(self.args.batch_size):
            x=self.samp_x[i,:,:,:,]
            y=self.samp_y[i,:,:,:,] 
            g_y=self.G(x.unsqueeze(0).cuda())
            axs[i, 0].imshow(to_pil_image(x))
            axs[i, 1].imshow(to_pil_image(y))
            axs[i, 2].imshow(to_pil_image(g_y.squeeze().cpu()))
        plt.show()  
        fig.savefig(self.args.save_path+title+'.png', bbox_inches='tight')
        fig.clf()

def parse_arguments(): 
     parser = HyperOptArgumentParser(strategy='grid_search')
     parser.add_argument('--num_epochs', type=int, default=2, help='The number of epochs to run') 
     parser.opt_list('--lrG', type=float,default=0.0002,tunable=True,options=[0.0001, 0.0002, 0.0003])
     parser.opt_list('--lrD', type=float,default=0.0002,tunable=True, options=[0.0001, 0.0002, 0.0003])
     parser.add_argument('--beta1', type=float,default=0.5)
     parser.add_argument('--beta2', type=float, default=0.999)
     parser.add_argument('--batch_size', type=int, default=3) 
     parser.add_argument('--lambd', type=float,default=100) 
     parser.add_argument('--mean', type=float, default=0.0)
     parser.add_argument('--std', type=float, default=0.02)
     parser.add_argument('--save_path', type=str, default='/default/pix2pix_results') 
     parser.add_argument('--path', type=str, default='/default/path') 
     return parser.parse_args()

def main(saved_w):
    args=parse_arguments() 
    for hparam_trial in args.trials(9):
        pix2pix=train_pix2pix(hparam_trial,saved_w)  
        my_model=pix2pix.run() 
    return my_model     

if __name__=='__main__':  
    my_model=main(False)
