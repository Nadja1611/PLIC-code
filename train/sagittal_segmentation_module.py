import os
os.chdir(".\\")
from functions import *
#https://github.com/ContinuumIO/anaconda-issues/issues/12194
#installation of tensorflow in prompt
#Python 3.8: conda install tensorflow-gpu=2.3 tensorflow=2.3=mkl_py38h1fcfbd6_0



os.chdir(".\\Data_npz")
data = np.load("Training_Babies.npz")
X = data["T1"]
Y = data["Labels"]
Labels=np.copy(Y)
indices = np.load("indices.npz")
indices = indices["indices"]
Thalamus = np.load("thalamus.npz")
'''Load Slice indices of Thalamus Label'''
thalamus = Thalamus["thalamus"]



os.chdir('.\\')
X = preprocessing_sagittal_crossvalidation(X,Y,indices,thalamus)[0]
Y=preprocessing_sagittal_crossvalidation(X,Y,indices,thalamus)[1]



X_train= generate_folds_sagittal(X)[0]
X_val= generate_folds_sagittal(X)[1]

Y_train= generate_folds_sagittal(Y)[0]
Y_val= generate_folds_sagittal(Y)[1]




  

#%%predict on coronal slices
os.chdir(".\weights")
for i in range(5):
    model4 = get_unet_sag_cor()
    model_checkpoint=ModelCheckpoint("weights_sagittal_cross"+str(i)+".hdf5",save_best_only=True,monitor='loss')
    h4 = model2.fit(X_train[i],Y_train[i], epochs=300, batch_size = 16, callbacks=[model_checkpoint], verbose=1, validation_data = ( X_val[i],Y_val[i]) )




#%%Reconstruction of the results, so that we can compare it to the results obtained on axial view

prediction_sag_noth=[]
for i in range(5):
    model4.load_weights("unet_weights_sagittal_cross"+str(i)+".hdf5")
    pred = model4.predict(X_val[i])
    pred = pred[:,:,:,0]
    prediction_sag_noth.append(pred)
prediction_sag_noth=np.concatenate(prediction_sag_noth,axis=0)
reconstruction_sag_noth=reconstruct_sag_crossval(prediction_sag_noth, Liste)



os.chdir(".\Data")

np.savez_compressed("Results_sagittal.npz", reconstruction_sag=reconstruction_sag_noth)


    
    


