

import os
os.chdir(".\\")
from functions import *

os.chdir(".\\Data")
data = np.load("Training_Babies.npz")
X = data["T1"]
Y = data["Labels"]
Labels=np.copy(Y)
indices = np.load("indices.npz")
indices = indices["indices"]

'''Load Slice indices of Thalamus Label'''
thalamus_training = Thalamus["L_train"]
thalamus_val = Thalamus["L_val"]


os.chdir(".\\")
X_train = preprocessing_coronal(X,Y,indices,thalamus_training,thalamus_val)[0]
Y_train=preprocessing_coronal(X,Y,indices,thalamus_training,thalamus_val)[1]
X_val=preprocessing_coronal(X,Y,indices,thalamus_training,thalamus_val)[2]
Y_val=preprocessing_coronal(X,Y,indices,thalamus_training,thalamus_val)[3]

X = np.concatenate((X_train,X_val),axis=0)
Y = np.concatenate((Y_train,Y_val),axis=0)

   

X_train= generate_folds_coronal(X)[0]
X_val= generate_folds_coronal(X)[1]

Y_train= generate_folds_coronal(Y)[0]
Y_val= generate_folds_coronal(Y)[1]



  

#%%predict on coronal slices
os.chdir(".\\weights")
for i in range(5):
    model4 = get_unet_sag_cor()
    model_checkpoint=ModelCheckpoint("weights_coronal_cross"+str(i)+".hdf5",save_best_only=True,monitor='loss')
    h4 = model2.fit(X_train[i],Y_train[i], epochs=300, batch_size = 16, callbacks=[model_checkpoint], verbose=1, validation_data = ( X_val[i],Y_val[i]) )




#%%Reconstruction of the results, so that we can compare it to the results obtained on axial view

prediction_cor_noth=[]
for i in range(5):
    model4.load_weights("unet_weights_coronal_cross"+str(i)+".hdf5")
    pred = model4.predict(X_val[i])
    pred = pred[:,:,:,0]
    prediction_cor_noth.append(pred)
prediction_cor_noth=np.concatenate(prediction_cor_noth,axis=0)
reconstruction_cor_noth=reconstruct_cor_crossval(prediction_cor_noth, Liste)



os.chdir(".\\Data")

np.savez_compressed("Results_coronal.npz", reconstruction_cor=reconstruction_cor_noth)


    
    

