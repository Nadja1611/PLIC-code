
import os

os.chdir(".\\")

from functions import *
os.chdir(".\\Data")
#%%we now read in the data obtained by application of plic-slice selector
data=np.load("Results_class_crossval.npz",allow_pickle=True)
lst=data["results"]
indi = data["indi"]
indi_val=[]
for i in range(0,5):
    indi_val.append(indi[i][0])


#%%we now train the axial segmentation model on the restricted volumes in axial view
os.chdir(".\\weights")
for i in range(5):
    model2=get_unet2()
    model_checkpoint=ModelCheckpoint("weights_axial_cross"+str(i)+".hdf5",save_best_only=True,monitor='loss')
    h4 = model2.fit(lst[i][2],lst[i][3][:,:,:,:1], epochs=300, batch_size = 16, callbacks=[model_checkpoint], verbose=1, validation_data = ( lst[i][0],lst[i][1][:,:,:,:1]) )


#%%predict and add data together again
prediction=[]
for i in range(5):
    model2.load_weights("weights_axial_cross"+str(i)+".hdf5")
    pred = model2.predict(lst[i][0])
    pred[pred[:,:,:,0]>0.5]=1
    pred[pred[:,:,:,0]<0.5] = 0
    pred = pred[:,:,:,0]
    pred = pred*mask
    prediction.append(pred)
    
'''reconstructs predicted patches of validation data into originial form'''    
reconstruction_axial = reconstruct_axial_crossval(prediction,indi_val) #shape (10854, 192, 192)
dice_a=evaluate_dice(reconstruction_axial,Labels)[0]
prediction_ax_noth=[]
for i in range(5):
    model2.load_weights("net2_crossvalidation"+str(i)+".hdf5")
    pred = model2.predict(lst[i][0])
    pred = pred[:,:,:,0]
    pred = pred*mask
    prediction_ax_noth.append(pred)
    
#reconstruction of predicted output masks of size 64 times 64 to original volumes     
reconstruction_ax_noth=reconstruct_axial_crossval(prediction_ax_noth, indi_val)
    
    
#%%save the results
#np.savez_compressed("Results_axial_crossvalidation.npz", Results=Result)
