# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 09:37:34 2023

@author: nadja
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import utils,models
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import PIL
from PIL import Image
from tqdm import trange
from time import sleep
from scipy.io import loadmat
import torchvision.datasets as dset
from torch.utils.data import sampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn import init, ReflectionPad2d, ZeroPad2d
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from unet_blocks import *
import numpy as np
import matplotlib.pyplot as plt
import gc

torch.manual_seed(0)
from functions_test import *


        
                

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class Autoencoder(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up_Autoencoder(1024, 512 // factor, bilinear))
        self.up2 = (Up_Autoencoder(512, 256 // factor, bilinear))
        self.up3 = (Up_Autoencoder(256, 128 // factor, bilinear))
        self.up4 = (Up_Autoencoder(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        logits = self.outc(x)
        return logits



class Segmentation_of_PLIC:
    def __init__(
        self,
        learning_rate: float = 5e-5,
        device: str = 'cuda:0',
        inputs: str = "T1",
        sliceview: str= "axial",
    ):
        self.mean_FP=[]
        self.mean_FN=[]
        self.Recall= 0
        self.max_mean_dice=0
        self.max_mean_dice_sag=0
        self.max_mean_dice_cor=0

        self.mean_recall=[]
        self.learning_rate = 5e-5
        self.net = UNet(1,1)
        self.weight = 0.001
        #self.inputs = "DWI"
        self.segmentation_net = UNet(n_channels=1,n_classes=1).to(device)
        self.segmentation_net_sag = UNet(n_channels=1,n_classes=1).to(device)
        self.segmentation_net_cor = UNet(n_channels=1,n_classes=1).to(device)
        
        self.optimizer = optim.Adam(self.segmentation_net.parameters(), lr=self.learning_rate)
        self.optimizer_cor = optim.Adam(self.segmentation_net_cor.parameters(), lr=self.learning_rate)
        self.optimizer_sag = optim.Adam(self.segmentation_net_sag.parameters(), lr=self.learning_rate)

        self.device = 'cuda:0'
        self.Dice = 0
        self.Dice_isles = 0
        self.inputs = inputs
        self.mean_dice=[]
        self.mean_dice_isles=[]
        self.mean_dice_isles_sag = []
        self.mean_dice_sag=[]
        self.mean_dice_isles_cor = []
        self.mean_dice_cor = []
        self.FP= 0
        self.FN = 0
        self.mean_spec=[]
        self.Spec = 0
        self.alpha=0.7
        self.gamma=0.5
        self.delta=0.5
        self.data_val_sag=None
        self.data_sag=None
        self.data_cor = None
        self.data_val_cor=None
        self.data = torch.ones((100,1000,100))
        
    def init_NW(self, device):
        if self.inputs == "T1":
            self.segmentation_net = UNet(n_channels=1,n_classes=1).to(device)    
            self.segmentation_net_cor = UNet(n_channels=1,n_classes=1).to(device)  
            self.segmentation_sag = UNet(n_channels=1,n_classes=1).to(device)       


        self.optimizer = optim.Adam(self.segmentation_net.parameters(), lr=self.learning_rate)
        self.optimizer_cor = optim.Adam(self.segmentation_net_cor.parameters(), lr=self.learning_rate)
        self.optimizer_sag = optim.Adam(self.segmentation_net_sag.parameters(), lr=self.learning_rate)


    def create_sag_cor_patches(self, X_new, Y_patch, indices):
        ''' generates the training patches for coronal and sagittal training'''
        X_train_sag = []
        Y_train_sag = []
        X_val_sag = []
        Y_val_sag = []
        X_train_cor = []
        Y_train_cor = []
        X_val_cor = []
        Y_val_cor = []
        for i in [50]:
            thalamus_train = list(np.ones(85)*i)
            thalamus_val = list(np.ones(12)*i)
            sag = preprocessing_sagittal(X_new,Y_patch,indices,thalamus_train,thalamus_val)
            cor = preprocessing_coronal(X_new,Y_patch,indices,thalamus_train,thalamus_val)
            Y_train_sag.append(np.moveaxis(sag[1][:,:,:,:1], 3, 1))      
            X_train_sag.append(np.moveaxis(sag[0], 3, 1))
            X_val_sag.append(np.moveaxis(sag[2], 3, 1))
            Y_val_sag.append(np.moveaxis(sag[3][:,:,:,:1], 3, 1))
            
            Y_train_cor.append(np.moveaxis(cor[1][:,:,:,:1], 3, 1))      
            X_train_cor.append(np.moveaxis(cor[0], 3, 1))
            X_val_cor.append(np.moveaxis(cor[2], 3, 1))
            print(cor[0].shape)
            Y_val_cor.append(np.moveaxis(cor[3][:,:,:,:1], 3,1 ))
           
        self.data_cor = torch.cat([torch.tensor(X_train_cor[0]), torch.tensor(Y_train_cor[0])], axis=1)
        self.data_cor_val = torch.cat([torch.tensor(X_val_cor[0]), torch.tensor(Y_val_cor[0])], axis=1)
        self.data_sag = torch.cat([torch.tensor(X_train_sag[0]), torch.tensor(Y_train_sag[0])], axis=1)
        self.data_sag_val = torch.cat([torch.tensor(X_val_sag[0]), torch.tensor(Y_val_sag[0])], axis=1)
        

    def create_sag_cor_patches_test(self, X_new,  indices):
        ''' generates the training patches for coronal and sagittal training'''
        X_train_sag = []
        Y_train_sag = []
        X_val_sag = []
        Y_val_sag = []
        X_train_cor = []
        Y_train_cor = []
        X_val_cor = []
        Y_val_cor = []
        thalamus_train = 50
        sag = preprocessing_sag_test(X_new, indices, thalamus_train)
        cor = preprocessing_cor_test(X_new, indices,thalamus_train)
        X_train_sag.append(sag)      

        
        X_train_cor.append(cor)
           
        self.data_cor = torch.tensor(X_train_cor[0])
        self.data_sag = torch.tensor(X_train_sag[0])
        
    def normalize(self,f): 
        '''normalize image to range [0,1]'''
        f = torch.tensor(f).float()
        f = (f-torch.min(f))/(torch.max(f)-torch.min(f))
        return f
    
    def standardise_coronal_sagittal(self,g): 
        '''normalize image to range [0,1]'''
        for i in range(len(g)):
            g[i][0] = torch.tensor(g[i][0]).float()
            g[i][0] = (g[i][0]-torch.mean(g[i][0]))
            g[i][0] = g[i][0]/torch.std(g[i][0])            
        return g

    def compute_weight(self):
        '''use the labels of the whole dataset and compute imbalance for BCE loss term'''
        shape = self.gt.shape
        shape_sag = self.data_sag.shape
        shape_cor = self.data_cor.shape
    
        self.weight = torch.sum(self.gt)/(shape[2]**2*shape[0])
        self.weight_cor = torch.sum(self.data_sag[:,1])/(shape[2]**2*shape[0])
        self.weight_sag = torch.sum(self.data_cor[:,1])/(shape[2]**2*shape[0])



    def segmentation_step(self,data):
        self.create_sag_cor_patches( X_new, Y_patch, indices)
        print(self.data_sag.dtype)
        Data_loader = DataLoader(self.data,
                                      batch_size=32, shuffle=True, num_workers=0, pin_memory=True)

        Data_loader_sag = DataLoader(self.data_sag,
                                      batch_size=32, shuffle=True, num_workers=0, pin_memory=True)

        Data_loader_cor = DataLoader(self.data_cor,
                                      batch_size=32, shuffle=True, num_workers=0, pin_memory=True)

        del(self.data)
        print(len(Data_loader))
        del(data)
      #  del(self.data_val)
     #   del(data_val)
        gc.collect()
        losses = []
        i = 0
        running_loss = .0
        for epoch in range(500):
            
            
            
            ''' segmentation of axial slices '''
            for features in Data_loader:
                #explanation: features[:,:1] is DWI, features[:,1:2] is ADC, features [:,2:3] gt
                input_x = features[:,0:1,:,:].to(self.device)
                segmentation_mask = features[:,1:2,:,:].to(self.device)                    
                self.optimizer.zero_grad()
                output = F.sigmoid(self.segmentation_net(input_x.float().to(self.device)))
                loss = self.dice_loss(segmentation_mask, output)
                loss.backward()
                self.optimizer.step()
                
                
                with torch.no_grad():
                    running_loss += loss.item()
                    if i %10 == 9:    
                        print('[Epoque : %d, iteration: %5d] loss: %.3f'%
                              (epoch + 1, i + 1, running_loss / 10))
                        running_loss = 0.0
                    i+=1 
            plt.subplot(1,3,1)
            plt.imshow((output[0][0]).detach().cpu())
            plt.subplot(1,3,2)
            plt.imshow(input_x[0][0].detach().cpu())
            plt.subplot(1,3,3)
            plt.imshow(segmentation_mask[0][0].detach().cpu())
            plt.show()          
                        
                
  
                

            '''   end of axial training '''
            '''---------------------------------- sagittal training ------------------- '''   
            for features in Data_loader_sag:
                #explanation: features[:,:1] is DWI, features[:,1:2] is ADC, features [:,2:3] gt
                input_x = features[:,0:1,:,:].to(self.device)
                segmentation_mask = features[:,1:2,:,:].to(self.device)
        
                    
                self.optimizer_sag.zero_grad()
                output = F.sigmoid(self.segmentation_net_sag(input_x.float().to(self.device)))
                loss_sag = self.dice_loss(segmentation_mask, output)
                #losses.append(loss.item())           
                loss_sag.backward()
                self.optimizer_sag.step()
                
                
                with torch.no_grad():
                    running_loss += loss_sag.item()
                    if i %10 == 9:    
                        print('[Epoque : %d, iteration: %5d] loss: %.3f'%
                              (epoch + 1, i + 1, running_loss / 10))
                        running_loss = 0.0
                    i+=1 
            plt.subplot(1,3,1)
            plt.imshow(torch.round(output[0][0].detach().cpu()))
            plt.subplot(1,3,2)
            plt.imshow(input_x[0][0].detach().cpu())
            plt.subplot(1,3,3)
            plt.imshow(segmentation_mask[0][0].detach().cpu())
            plt.show()          
                    
                
      
                                   

            ''''''''''''''''''''' end of sagittal training -----------------------------'''
            
            ''' coronal training '''   
            for features in Data_loader_cor:
                #explanation: features[:,:1] is DWI, features[:,1:2] is ADC, features [:,2:3] gt
                input_x = features[:,0:1,:,:].to(self.device)
                segmentation_mask = features[:,1:2,:,:].to(self.device)
        
                    
                self.optimizer_cor.zero_grad()
                output = F.sigmoid(self.segmentation_net_cor(input_x.float().to(self.device)))
                loss_cor = self.dice_loss(segmentation_mask, output)
                #losses.append(loss.item())           
                loss_cor.backward()
                self.optimizer_cor.step()
                
                
                with torch.no_grad():
                    running_loss += loss_cor.item()
                    if i %10 == 9:    
                        print('[Epoque : %d, iteration: %5d] loss: %.3f'%
                              (epoch + 1, i + 1, running_loss / 10))
                        running_loss = 0.0
                    i+=1 
            plt.subplot(1,3,1)
            plt.imshow(torch.round(output[0][0].detach().cpu()))
            plt.subplot(1,3,2)
            plt.imshow(input_x[0][0].detach().cpu())
            plt.subplot(1,3,3)
            plt.imshow(segmentation_mask[0][0].detach().cpu())
            plt.show()          
                
  
            

                #self.free_gpu_cache()  



            
########### tversky loss ####################################            
    def tversky(self, tp, fn, fp):
        loss2 = 1- ((torch.sum(tp)+0.000001)/((torch.sum(tp) + self.gamma*torch.sum(fn) + self.delta*torch.sum(fp)+0.000001)))
        return loss2
    

    def evaluate(self, segmentation_mask, output):
        output = torch.round(output)
        tp = torch.sum(output*segmentation_mask)
        tn = torch.sum((1-output)*(1-segmentation_mask))
        fn = torch.sum((1-output)*(segmentation_mask))
        fp = torch.sum((output)*(1-segmentation_mask))
        Recall = (tp+0.0001)/(tp + fn + 0.0001)
        Spec = (tn / (tn + fp))
        Dice = 2*tp/(2*tp + fn + fp)
        #AP = AP_score(np.asarray(segmentation_mask.detach().cpu()), np.asarray(output.detach().cpu()), 0.5, 0.5)
        #S_score = Sens_score(np.asarray(segmentation_mask.detach().cpu()), np.asarray(output.detach().cpu()), 0.5, 0.5)
        #F_score = FP_score(np.asarray(segmentation_mask.detach().cpu()), np.asarray(output.detach().cpu()), 0.5, 0.5)
        im1 = np.asarray(segmentation_mask.detach().cpu()).astype(bool)
        im2 = np.asarray(output.detach().cpu()).astype(bool)

        if im1.shape != im2.shape:
            raise ValueError(
                "Shape mismatch: im1 and im2 must have the same shape.")

        im_sum = im1.sum() + im2.sum()
        if im_sum == 0:
            return 1.0

        # Compute Dice coefficient
        intersection = np.logical_and(im1, im2)

        dice_isles = 2.0 * intersection.sum() / im_sum

        self.Dice_isles += dice_isles
        self.Dice += Dice.detach().cpu()
        self.FP += fp.detach().cpu()
        self.FN += fn.detach().cpu()
        self.Recall += Recall
        self.Spec += Spec
        # self.AP += AP
        # self.F_score += F_score
        # self.S_score += S_score
        del(fn, fp, tn, tp)
        gc.collect()



###### loss for joint reconstruction and segmentation ########
    def dice_loss(self, segmentation_mask, output):        
        weights = torch.stack([torch.tensor(1-self.weight), torch.tensor(self.weight)]).to(self.device)
        output = torch.stack([output, 1-output], axis=-1)
        segmentation_mask = torch.stack([segmentation_mask, 1-segmentation_mask], axis=-1)

       # weights = 1-torch.tensor(self.weight)
        output = torch.clip(output, min = 1e-6)
        loss1 = -torch.sum(segmentation_mask * torch.log(output)* weights,axis = -1)
        loss1 = torch.mean(loss1)
        '''tversky preperation'''
        y_true_f = torch.flatten(segmentation_mask[:,:,:,:])
        y_pred_f = torch.flatten(output[:,:,:,:])
        fp = (1-y_true_f)*y_pred_f
        fn = (1-y_pred_f)*y_true_f
        tp = y_pred_f*y_true_f

        loss = (self.alpha*loss1) + (1-self.alpha)*(self.tversky(tp,fn,fp)) #+ recon_weight*(loss_rec + loss_rec2) + n*sym_loss
        #loss = self.tversky(tp,fn,fp
        del(tp, fp, fn, y_true_f, y_pred_f)
        gc.collect()
        return loss
    

#mynet.segmentation_step(data, data_val)

##################### predict for alle three directions ##############################
def predict( mynet, weight_path):
    Pred_cor=[]
    Pred_sag=[]
    Pred = []
    Image  =[]
    Image_cor = []
    Image_sag = []
    
    ''' load the weights for the three nw '''
    mynet.segmentation_net.load_state_dict(torch.load(weight_path + "best_dice_weights_T1.hdf5"))
    mynet.segmentation_net_sag.load_state_dict(torch.load(weight_path + "best_dice_weights_sag_T1.hdf5"))
    mynet.segmentation_net_cor.load_state_dict(torch.load(weight_path + "best_dice_weights_cor_T1.hdf5"))

    ''' create the dataloaders '''
    Data_loader_val = DataLoader(mynet.data,
                                  batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    Data_loader_val_sag = DataLoader(mynet.data_sag,
                                 batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    Data_loader_val_cor = DataLoader(mynet.data_cor,
                                 batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    '''load the T1 images and create the predictions'''
    for features in Data_loader_val:
        input_x = features[:,0:1,:,:].to(mynet.device)
        segmentation_mask = features[:,1:2,:,:].to(mynet.device)          
        output = F.sigmoid(mynet.segmentation_net(input_x.float().to(mynet.device)))
        Pred.append(output.detach().cpu())
        Image.append(input_x.detach().cpu())
    for features in Data_loader_val_sag:
        input_x = features[:,0:1,:,:].to(mynet.device)
        segmentation_mask = features[:,1:2,:,:].to(mynet.device)          
        output_sag = F.sigmoid(mynet.segmentation_net_sag(input_x.float().to(mynet.device)))
        Pred_sag.append(output_sag.detach().cpu())
        Image_sag.append(input_x.detach().cpu())

    for features in Data_loader_val_cor:
        input_x = features[:,0:1,:,:].to(mynet.device)
        segmentation_mask = features[:,1:2,:,:].to(mynet.device)          
        output_cor = F.sigmoid(mynet.segmentation_net_cor(input_x.float().to(mynet.device)))    
        Pred_cor.append(output_cor.detach().cpu())
        Image_cor.append(input_x.detach().cpu())

    Cor =  torch.cat(Pred_cor,axis=0) 
    Sag =  torch.cat(Pred_sag,axis=0) 
    Ax =  torch.cat(Pred,axis=0) 
    Im_ax =  torch.cat(Image,axis=0) 
    Im_cor =  torch.cat(Image_cor,axis=0) 
    Im_sag =  torch.cat(Image_sag,axis=0) 


    return Ax, Sag, Cor, Im_ax, Im_cor, Im_sag



''' predict on axial, sagittal and coronal views and reconstruct coronal and sagittal into axial views and combine to output '''
def predict_and_compute_volume(mynet, weight_path, indices, Voxels, Spacings, T1_orig):
    Sag = []
    Cor = []
    Combi=[]
    Ax=[]
    Indi = []
    sag = reconstruct_sag(predict(mynet,weight_path)[1][:64].squeeze(1), indices[0],50)
    Sag.append(sag)
    cor = reconstruct_cor(predict(mynet,weight_path)[2][:64].squeeze(1), indices[0],50)
    Cor.append(cor)
    ax = reconstruct_axial(predict(mynet,weight_path)[0].squeeze(1),indices[0],0)
    Ax.append(ax)
    Combi.append(cor + sag + ax )
    Combi = np.asarray(Combi)
    Comb=Combi>1.2   
    if list(Comb[0].shape) != Voxels:
        print("yes")
        Comb = np.round(resize(Comb[0], (Voxels[0], Voxels[1], Voxels[2])))
        print(Comb.shape)
        Volume = np.sum(Comb) * Spacings[0]*Spacings[1]*Spacings[2]
        print(Volume)

    else:
        Comb = Comb[0]
        Volume = np.sum(Comb) * Spacings[0]*Spacings[1]*Spacings[2]
    return Volume, Comb, T1_orig    
