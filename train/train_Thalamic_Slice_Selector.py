# -*- coding: utf-8 -*-
"""
Created on Mon May 23 18:09:59 2022

@author: Nadja Gruber
"""

import os
os.chdir(".\\")
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from functions import *
#from net1 import ind_test,ind_train, Xseg_test, Xseg_train, Lab_thal_train,Lab_thal_test

os.chdir(".\\Data")

data = np.load("Training_Babies.npz",allow_pickle=True)
patients= data["T1"]
indices = data["indices"]

#now we load the results which were obtained by the PLIC slice selector network
data2 = np.load("Results_class_crossval.npz", allow_pickle=True)
#ind_test corresponds to the indices that were predicted on the validation set to be a slice containing plic
ind_test = data2["indi"][:,0]
ind_test = np.concatenate(ind_test, axis=0)
#lst contains the predictions obtained by PLIC-Slice-Selector, so only the relvant slices, as well as the Thalamus labels 
#only restricted on these slices
lst = data2["results"]

 


#construct the kernel for convolving the binary labels of thalamic slice, see paper 2.2.2
sigma = 0.22
x = np.linspace(-1,1,10)
gaussi = np.exp(-x**2/sigma)




#now we train the thalamic slice selector on the 5 folds
os.chdir(".\\weights")

for i in range(5):
    model3 = Thalamic_Slice_Selector()
    label_test = np.convolve(lst[i][5][:,0],gaussi,'same')
    label_train = np.convolve(lst[i][4][:,0],gaussi,'same')
    model3.fit(lst[i][2],label_train , epochs=20, validation_data=(lst[i][0],label_test))
    model3.save_weights("thalamus_crossval"+str(i)+".hdf5")

#we use these weights now for predictions
P=[]
for i in range(5):
    model3.load_weights("thalamus_crossval"+str(i)+".hdf5")
    predictions = model3.predict(lst[i][0])
    P.append(predictions)
    
#reconstruct the results,  Identification of boundary between upper and middle
#third of the thalamus
P=np.concatenate(P,axis=0)
reconstruction= reconstruct_thal(np.asarray(P),ind_test,indices)
Results=reconstruction[2]

os.chdir(".\\Data")
#We save the indices, and use them for sagittal and coronal training to get the slices with center slice at correct level
np.savez_compressed("thalamus.npz",thalamus = Results)



