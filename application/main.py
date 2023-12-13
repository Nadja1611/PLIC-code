
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:23:55 2023

@author: Nadja Gruber and Martin Knoflach
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
import SimpleITK as simpli 
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
from fpdf import FPDF 

import numpy as np
import matplotlib.pyplot as plt
import gc
from read_in_DICOM import *
torch.manual_seed(0)
from functions_test import *
from predict_test import *
import argparse
from pydicom.datadict import dictionary_VR

VERBOSE = 1
# ----

parser = argparse.ArgumentParser(description='Arguments for segmentation network.',add_help=False)
parser.add_argument('--outputdir', type=str, 
                    help='directory for outputs', default= "D://PLIC Segmentation//PLIC_pytorch//")
parser.add_argument('--inputdir', type=str, 
                    help='directory for input files', default = "D://PLIC Segmentation//PLIC_pytorch//Babies//" )
parser.add_argument('--weightdir', type=str, 
                    help='directory for weights', default = "D://PLIC Segmentation//PLIC_pytorch//" )
parser.add_argument('--Patient', type = int, help="which Patient do we want to be segmented", default = 23)
parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                    help='This python application computes the volume of the posterior limb of the internal capsule of Babies on T1 MRI, and saves a txt file')
args = parser.parse_args()

weight_path = args.weightdir
'''create the data '''
T1 = read_in(args.inputdir,args.Patient)[0]
T1_orig = read_in(args.inputdir,args.Patient)[5]
X_new = read_in(args.inputdir,args.Patient)[1]
indices = read_in(args.inputdir,args.Patient)[2]
Voxels = read_in(args.inputdir,args.Patient)[3]
Spacings = read_in(args.inputdir,args.Patient)[4]
header = read_in(args.inputdir,args.Patient)[6]
if header.StudyDescription == "Schädel":
    X_new = np.expand_dims(X_new, axis=-1)
    X_new = np.moveaxis(X_new, 3,1)
    X_new = torch.tensor(X_new)
    
    data = torch.tensor(X_new)
    
    mynet = Segmentation_of_PLIC(inputs = "T1")
    mynet.data= data
    '''create the coronal and sagittal data '''
    mynet.create_sag_cor_patches_test( X_new, indices[0])
    mynet.standardise_coronal_sagittal(mynet.data_sag)
    mynet.standardise_coronal_sagittal(mynet.data_cor)
    '''compute weights for BCE '''
    mynet.init_NW(device=mynet.device) 
    
    ''' compute volume in mm^3 '''
    result=predict_and_compute_volume(mynet, weight_path, indices, Voxels, Spacings, T1_orig)
    volume = result[0]
    '''- predicted segmentation mask resized to original size '''
    segmented_image = result[1]
    ''' original T1 before resizing '''
    original_T1 = result[2]
    
    os.chdir(args.outputdir)
    lines = ["Computed PLIC volume",'Baby: '+ str(header.PatientID), 'Date: '+str(header.StudyDate),'Time: '+str(header.StudyTime), 'Age: '+ str(header.PatientAge),'Sex: '+ str(header.PatientSex), 'Description: '+header.StudyDescription, 'Volume:'+ str(np.round(volume,3)) + "mm^3"]
    with open('volume_'+str(header.PatientID)+'.txt', 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')
    
    pdf = FPDF()      
    # Add a page 
    pdf.add_page()  
    # set style and size of font  
    # that you want in the pdf 
    pdf.set_font("Arial", size = 10)
    # open the text file in read mode 
    f = open('volume_'+str(header.PatientID)+'.txt', "r") 
    # insert the texts in pdf 
    for x in f: 
        pdf.cell(50,5, txt = x, ln = 1, align = 'L') 
    # save the pdf with name .pdf 
    pdf.output('volume_'+str(header.PatientID)+".pdf")
else:
    print("Error, wrong region!")    