# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 11:10:19 2023

@author: nadja
"""

from skimage import io
import numpy as np
from matplotlib import pyplot, cm
from matplotlib.pyplot import imshow
import nibabel as nib
import glob
from matplotlib.pyplot import imshow, colorbar
import argparse
from matplotlib import pyplot as plt


from PIL import ImageEnhance
from sklearn.metrics import roc_curve, auc
import pydicom
import pandas as pd
import os
from os import listdir
from PIL import Image
from skimage.transform import resize

'''reads in labels and data, saves indices i.e. number of slices, as well as T1 images in a row'm in total'''
''''we work with 98 babies. 3 are excluded'''



input_path = "C://Users//nadja//Documents//PLIC Segmentation//PLIC_pytorch//Babies//"
'''--------------------------------read in DWI----------------------------------------'''
def read_in(input_path):
    ds=[]
    Patient=[]
    Patches = []
    for i in range(100,200):   
        PathDicom_t1 =  input_path + "P" +str(i)+"//T1//DCM0"
        for j in os.listdir(PathDicom_t1): 
            if j != "F0000000":
                ds1 = pydicom.dcmread( input_path + "P"+str(i)+"//T1//DCM0//"+j)               
                ds1 = ds1.pixel_array
                
                ds.append(ds1)  
        ds = np.asarray(ds)        
        if ds.shape[0] > 110 or ds.shape[0]<90:
            print(ds.shape)
            print("yes")
            ds = resize(ds, (100, 192,192))
    
        else:
            print(ds.shape)
            ds = resize(ds, (np.shape(ds)[0], 192,192))    
    
            
        Patient.append(np.asarray(ds))
    
        Patches.append(np.asarray(ds)[:,65:129,65:129])
    
        ds=[]
    Patches = np.asarray(np.concatenate(Patches,axis=0))    
        
    for i in range(len(Patches)):
        Patches[i] = Patches[i] - np.mean(Patches[i])
        Patches[i] = Patches[i]/np.std(Patches[i])    
    #for i in [x for x in range(358,366)]:
    T1=np.concatenate(Patient, axis=0)
    T1=np.asarray(T1)
    
    indices = []
    for i in range(len(Patient)):
        length = np.shape(Patient[i])[0]
        indices.append(length)
    
    
    
    
    '''Grundstruktur und slices sortiert lassen'''
    liste = []
    liste.append(T1[:indices[0]])
    for i in range(len(Patient)-1):
        index1 = int(np.sum(indices[:i]))
        index2 = int(np.sum(indices[:i])+indices[i+1])
        P = T1[index1:index2]
        P=np.asarray(P)
        liste.append(P)
    
    Patienten_liste = liste
    return T1, Patches, indices


np.savez_compressed("Babies.npz",  T1 = read_in(input_path)[0], indices = read_in(input_path)[2], T1_patches = read_in(input_path)[1])



