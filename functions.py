# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 12:35:03 2021

@author: nadja
"""
import os
import tensorflow as tf
gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D,Dropout, Input, concatenate, Conv2DTranspose
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import layers
from scipy import ndimage, misc
from skimage.measure import label, regionprops, regionprops_table

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import os
import matplotlib.pylab as plt
from skimage.transform import rescale, resize, downscale_local_mean

'''standardizes the data and makes patches of size 64x64 out of original images'''
def preprocessing(X):
    X = X[:,65:129,65:129]
    liste=[]
    for i in range(len(X)):
        m = (X[i]-np.mean(X[i]))/(np.std(X[i])+0.0001)
        liste.append(m)
    X = np.asarray(liste)
    X=np.expand_dims(X,axis=-1)
    return X


def normalizing(X):
    liste=[]
    for i in range(len(X)):
        m = (X[i]-np.mean(X[i]))/(np.std(X[i])+0.0001)
        liste.append(m)
    X = np.asarray(liste)
    return X

def augmenting(X,angle):
    liste=[]
    for i in range(len(X)):
        m =  ndimage.rotate(X,angle)
        liste.append(m)
    X = np.asarray(liste)
    X=np.expand_dims(X,axis=-1)
    return X


def create_boxes_cor(X,levels,indices):
    Matrix = np.zeros_like(X)
    Images=[]
    for i in range(len(levels)):
        level= levels[i]+ np.sum(indices[:i])
        Matrix[65:129,int(level)-12:int(level)+12,65:129]=1
        image = X[68:124,int(level)-28:int(level)+28,68:124]
        Images.append(image)  
    Images=np.asarray(Images)
    return Matrix,Images



def create_boxes_sag(X,levels,indices):
    Matrix = np.zeros_like(X)
    Images=[]
    for i in range(len(levels)):
        level= levels[i]+ np.sum(indices[:i])
        Matrix[65:129,int(level)-12:int(level)+12,65:129]=1
        image = X[68:124,int(level)-28:int(level)+28,68:124 ]
        Images.append(image)  
    Images=np.asarray(Images)
    return Matrix,Images



def generalized_dice_loss(weight_bg,weight_fg):
    def custom_loss(y_true, y_pred):
        y_true_fg = y_true[:,:,:,0]
        y_pred_fg = y_pred[:,:,:,0]
        y_true_bg = 1-y_true[:,:,:,0]
        y_pred_bg = 1-y_pred[:,:,:,0]
        eps = 1e-6

        Ncl = 2 #we do binary segmentation, so foreground and background class
        weights = 1. / (tf.stack([weight_fg,weight_bg])** 2)
        w = tf.where(tf.math.is_finite(weights), weights, eps)
    #compute generalized dice
        numerator = w[0]*tf.reduce_sum(y_true_fg*y_pred_fg) + w[1]*tf.reduce_sum(y_true_bg*y_pred_bg)
        denominator = w[0]*tf.reduce_sum(y_true_fg+y_pred_fg) + w[1]*tf.reduce_sum(y_true_bg+y_pred_bg)
    
        gen_dice_coef = 2*numerator/denominator
        loss = 1-gen_dice_coef
        return loss
    return custom_loss


'''put back sagittal results to original sliced images of size 192x192'''
def reconstruct_sag(X_org,X, indices,levels):
    liste = []
#liste.append(T1[:indices[0]])
    for i in range(len(indices)):
        index1 = int(np.sum(indices[:i]))
        index2 = int(index1 + indices[i])
        P = X_org[index1:index2]
        P=np.asarray(P)
        liste.append(P)
    liste = np.asarray(liste)   
    Matrix=[]
    for i in range(len(liste)):
        M = np.zeros_like(np.moveaxis(liste[i],2,0))
        M=M.astype("float32")
        Matrix.append(M)
   # Matrix = np.asarray(Matrix)    
    for j in range(len(levels)):
        index1 = int(j*56)
        index2 = int(j*56 +56)
        Matrix[j][68:124,int(levels[j])-28:int(levels[j])+28,68:124]=X[index1:index2,:,:,0]
    return Matrix
    


'''put back coronal results to original sliced images of size 192x192'''
def reconstruct_cor(X_org,X, indices,levels):
    liste = []
#liste.append(T1[:indices[0]])
    for i in range(len(indices)):
        index1 = int(np.sum(indices[:i]))
        index2 = int(index1 + indices[i])
        P = X_org[index1:index2]
        P=np.asarray(P)
        liste.append(P)
    liste = np.asarray(liste)   
    Matrix=[]
    for i in range(len(liste)):
        M = np.zeros_like(np.moveaxis(liste[i],0,1))
        M=M.astype("float32")
        Matrix.append(M)
   # Matrix = np.asarray(Matrix)    
    for j in range(len(levels)):
        index1 = int(j*56)
        index2 = int(j*56 +56)
        Matrix[j][68:124,int(levels[j])-28:int(levels[j])+28,68:124]=X[index1:index2,:,:,0]
    return Matrix
    
 
def reconstruct_test(pred,ind,indices):
    k = np.zeros((np.sum(indices),192,192,1))
    for i in range(0,np.shape(pred)[0]):
        p=resize(pred[i], (64,64))
        k[ind[i],65:129,65:129,0] = p[:,:]
    return k


def preprocessing_coronal(X,Y,indices,thalamus_training,thalamus_val):
    X_train = np.moveaxis(X[:np.sum(indices[:85])],0,1)
    Y_train = np.moveaxis(Y[:np.sum(indices[:85])],0,1)
    X_val = np.moveaxis(X[np.sum(indices[:85]):],0,1)
    Y_val = np.moveaxis(Y[np.sum(indices[:85]):],0,1)
    X_res = create_boxes_cor(X_train,thalamus_training,indices[:85])[1]
    Y_res = create_boxes_cor(Y_train,thalamus_training,indices[:85])[1]
    X_res = preprocessing_no_patching(X_res)
    X_res_val = create_boxes_cor(X_val,thalamus_val,indices[85:])[1]
    Y_res_val = create_boxes_cor(Y_val,thalamus_val,indices[85:])[1]
    X_res_val = preprocessing_no_patching(X_res_val)
    X_train=np.concatenate(X_res,axis=0)
    X_val=np.concatenate(X_res_val,axis=0)
    Y_train=np.concatenate(Y_res,axis=0)
    Y_val=np.concatenate(Y_res_val,axis=0)
    Y_train = np.expand_dims(Y_train, axis=-1)
    Y_val = np.expand_dims(Y_val, axis=-1)
    return X_train,Y_train,X_val,Y_val

'''preprocessing sagittal data for training'''
def preprocessing_sagittal(X,Y,indices,thalamus_training,thalamus_val):
    X_train = np.moveaxis(X[:np.sum(indices[:85])],2,0)
    Y_train = np.moveaxis(Y[:np.sum(indices[:85])],2,0)
    X_val = np.moveaxis(X[np.sum(indices[:85]):],2,0)
    Y_val = np.moveaxis(Y[np.sum(indices[:85]):],2,0)
    X_res = create_boxes_sag(X_train,thalamus_training,indices[:85])[1]
    Y_res = create_boxes_sag(Y_train,thalamus_training,indices[:85])[1]
    X_res = preprocessing_no_patching(X_res)
    X_res_val = create_boxes_sag(X_val,thalamus_val,indices[85:])[1]
    Y_res_val = create_boxes_sag(Y_val,thalamus_val,indices[85:])[1]
    X_res_val = preprocessing_no_patching(X_res_val)
    X_train=np.concatenate(X_res,axis=0)
    X_val=np.concatenate(X_res_val,axis=0)
    Y_train=np.concatenate(Y_res,axis=0)
    Y_val=np.concatenate(Y_res_val,axis=0)
    Y_train = np.expand_dims(Y_train, axis=-1)
    Y_val = np.expand_dims(Y_val, axis=-1)
    return X_train,Y_train,X_val,Y_val



'''preprocessing of test data for prediction of coronal'''
def preprocessing_cor_test(test,indices_test2,thalamus):
    X_train = np.moveaxis(test,0,1)
    X_res = create_boxes_sag(X_train,thalamus,indices_test2)[1]
    X_res = preprocessing_no_patching(X_res)
    X_train=np.concatenate(X_res,axis=0)
    return X_train


'''preprocessing of test data for prediction of sagittal'''
def preprocessing_sag_test(test,indices_test2,thalamus):
    X_train = np.moveaxis(test,2,0)
    X_res = create_boxes_sag(X_train,thalamus,indices_test2)[1]
    X_res = preprocessing_no_patching(X_res)
    X_train=np.concatenate(X_res,axis=0)
    return X_train


'''standard preprocessing, i.e. only standardization and expansion of dimensions'''
def preprocessing_no_patching(X):
    liste=[]
    for i in range(len(X)):
        m = (X[i]-np.mean(X[i]))/np.std(X[i])
        liste.append(m)
    X = np.asarray(liste)
    X=np.expand_dims(X,axis=-1)
    return X


    
def reconstruct(pred,ind,indices):
    k = np.zeros((np.sum(indices[85:]),192,192,1))
    for i in range(0,np.shape(pred)[0]):
        p=resize(pred[i], (64,64))
        k[ind[i],65:129,65:129,0] = p[:,:]
    return k

def reconstruct_test(pred,ind,indices):
    k = np.zeros((np.sum(indices),192,192,1))
    for i in range(0,np.shape(pred)[0]):
        p=resize(pred[i], (64,64))
        k[ind[i],65:129,65:129,0] = p[:,:]
    return k

def reconstruct_thal(pred,ind,indices):
    k = np.zeros((np.sum(indices),1))
    for i in range(0,np.shape(pred)[0]):
        p=pred[i]
        k[ind[i],:] = p[:]
    Result=[]
    maximum=[]
    for i in range(len(indices)):
        lower = np.sum(indices[:i])
        upper = np.sum(indices[:i])+indices[i]
        R = k[lower:upper]
        maxi = np.argmax(R)
        Result.append(R)  
        maximum.append(maxi)
    Results=np.asarray(Result)    
    return k,Results,maximum


def make_patient_list(indices):
    os.chdir("C://Users//nadja//Documents//PLIC_programm/Data_npz")
    data = np.load("Training_Babies.npz",allow_pickle=True)
    patients= data["T1"]
    indices = data["indices"]

    liste = []
#liste.append(T1[:indices[0]])
    for i in range(len(indices)):
        index1 = int(np.sum(indices[:i]))
        index2 = int(index1 + indices[i])
        P = patients[index1:index2]
        P=np.asarray(P)
        liste.append(P)
    return liste,indices

def make_labels_list(indices):
    os.chdir("C://Users//nadja//Documents//PLIC_programm/Data_npz")
    data = np.load("Training_Babies.npz",allow_pickle=True)
    patients= data["Labels"]
    indices = data["indices"]

    liste = []
#liste.append(T1[:indices[0]])
    for i in range(len(indices)):
        index1 = int(np.sum(indices[:i]))
        index2 = int(index1 + indices[i])
        P = patients[index1:index2]
        P=np.asarray(P)
        liste.append(P)
    return liste,indices
#Segmentierungen patientenweise
def make_patient_list_results(indices,data):
#liste.append(T1[:indices[0]])
    liste=[]
    for i in range(len(indices)):
        index1 = int(np.sum(indices[:i]))
        index2 = int(index1 + indices[i])
        P = data[index1:index2]
        P=np.asarray(P)
        liste.append(P)
    return liste




def show_final_results(Results,indices):    
    images=[]
    for i in range(len(Results)):
        image=make_patient_list(indices)[0][1][Results[i]]
        images.append(image)
    return images   
    
######
from keras import backend as K
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

def dice_loss(y_true, y_pred):
  y_true = tf.cast(y_true, tf.float32)
 # y_pred = tf.sigmoid(y_pred[:,:,:,0])
  numerator = 2 * tf.reduce_sum(y_true[:,:,:,0] * y_pred[:,:,:,0])
  denominator = tf.reduce_sum(y_true[:,:,:,0] + y_pred[:,:,:,0]+ K.epsilon())
  return 1 - numerator / denominator


'''applied windowing'''
def contrast(im):
    mini = np.zeros(np.shape(im)[0])
    maxi = np.zeros(np.shape(im)[0])

    for i in range(np.shape(im)[0]):
        mini[i] = np.percentile(im[i][im[i]>0],35)
        im[i] = (im[i] -mini[i])
        maxi[i] = np.max(im[i])+0.0001
        im[i][im[i]<0] = 0
        im[i] = im[i]/maxi[i]
    return im

'''predictions of classifier on validation set'''
def predict_classifier(X,Y,indices,predictions,threshold):
    index = predictions
    ind = []
    for i in range(0, np.shape(X)[0]):
        if index[i,0] > threshold:
            ind.append(i)
    ind = cluster(ind)        
    Xseg = [X[i] for i in ind]      #construct training data for unet   
    Xseg = np.asarray(Xseg)
    Yseg = [Y[i] for i in ind]
    Yseg = np.asarray(Yseg)  
    return Xseg, Yseg, ind     
            

'''get rid of false negatives, if slices at beginning or at the end are 
classified as false positives, they are clustered and deleted. a clsuter of size
smaller than K is removed'''
def cluster(items_list):
    cluster_liste=[]
    items = sorted(items_list)
    clusters = [[items[0]]]
    for item in items[1:]:
        cluster = clusters[-1]
        last_item = cluster[-1]
        if item - last_item <3:
            cluster.append(item)
        else:
            clusters.append([item])
    data2 = [i for i in clusters if len(i)<7]
    for items in data2:
        clusters.remove(items)
    cluster_liste.append(clusters)    
    cluster_liste=[item for sublist in cluster_liste for item in sublist]  
    cluster_liste=[item for sublist in cluster_liste for item in sublist]       
    return cluster_liste


'''predictions of classifier on test set, avoids false positive
slices at beginning and end via cluster function'''
def predict_classifier_test(X,indices,predictions,threshold):
    index = predictions
    ind = []
    for i in range(0, np.shape(X)[0]):
        if index[i,0] > threshold:
            ind.append(i)
    ind = cluster(ind)        
    Xseg = [X[i] for i in ind]      #construct training data for unet   
    Xseg = np.asarray(Xseg) 
    return Xseg,  ind  

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss         
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

# 'function for plotting ROC curve and find optimal threshold'
from sklearn.metrics import roc_auc_score
from sklearn import metrics
def ROC(X,Y,predictions):
     def plot_roc_curve(fpr, tpr):
         plt.plot(fpr, tpr, color='orange', label='ROC')
         plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
         plt.xlabel('False Positive Rate')
         plt.ylabel('True Positive Rate')
         plt.title('Receiver Operating Characteristic (ROC) Curve')
         plt.legend()
         plt.show()
    
     y_true = np.array(Y[:,0].flatten())
     y_scores = np.array(predictions[:,0].flatten())
     fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)

     print(roc_auc_score(y_true, y_scores))
     optimal_idx = np.argmax(tpr-fpr)
     optimal_threshold = thresholds[optimal_idx]
     print("Threshold value is:", optimal_threshold)
     plot_roc_curve(fpr, tpr)
     print(optimal_idx)
     return optimal_threshold




'''to avoid false negatives lying outside plic region, multiplies 
the whole image with a small box of ones'''
def restriction_box(data):
    #one_matrix = np.zeros_like(data[0][0])   
    #one_matrix[70:125,70:125]=1 
    mask_matrix=np.zeros_like(data[0][0])
    mask_matrix[85:115,70:92]=1
    mask_matrix[85:115,100:122]=1
    one_matrix = mask_matrix.astype('int8')
    refinement=[]
    for i in range(len(data)):
        p = []
        for j in range(len(data[i])):
            f = data[i][j]*one_matrix
            p.append(f)
        refinement.append(p)
    data=np.asarray(refinement) 
    return data




#############################not used######################
'''the results of model1 are rewritten to patientwise form'''
def delete_FP(indices,data):
    liste=[]
    for i in range(len(indices)):
        index1 = int(np.sum(indices[:i]))
        index2 = int(index1 + indices[i])
        data2 = [i for i in data if i >= index1 and i<= index2]
        liste.append(data2)
    return liste


from skimage.measure import regionprops

'''left and right area of plic on thalamuslevel , as well as LAT and AP and total Volumes'''
'''left and right area of plic on thalamuslevel , as well as LAT and AP and total Volumes'''
def area_leftright_total_diam1(Final_Results,Results):
    areas=[]
    areas_short=[]
    for i in range(len(Results)):
        left=np.sum(Final_Results[i][Results[i]][85:115,70:92])
        right=np.sum(Final_Results[i][Results[i]][85:115,100:122])
        if left == 0:
            width_l=height_l = 0
        if right ==0:
            width_r=height_r = 0
        else:
            properties_left=regionprops(Final_Results[i][Results[i]][85:115,70:92][:,:,0])
            properties_right=regionprops(Final_Results[i][Results[i]][85:115,100:122][:,:,0])
            for p in properties_left:
                min_row_l, min_col_l, max_row_l, max_col_l = p.bbox
                width_l = max_col_l-min_col_l    
                height_l=max_row_l-min_row_l
                for p in properties_right:
                    min_row_r, min_col_r, max_row_r, max_col_r = p.bbox  
                width_r = max_col_r-min_col_r    
                height_r=max_row_r-min_row_r 
        Volume=0
        for j in range(Results[i]-7,Results[i]+7):
            Volume1 = np.sum(Final_Results[i][j])
            Volume=Volume+Volume1
            
        areas.append(["Baby"+str(i+71),"PLIC_l="+str(right),"PLIC_r="+str(left),"total="+str(left+right),
                      "LAT_l="+str(width_r),"AP_l="+str(height_r),"LAT_r="+str(width_l),"AP_r="+
                      str(height_l), "Vol="+str(Volume)])
        areas_short.append(["Baby"+str(i+71),right,left,left+right,width_r,height_r,width_l,height_l,Volume,len(Final_Results[i])])
    return areas,areas_short





def area_leftright_total_diam2(Final_Results,Results):
    areas=[]
    areas_short=[]
    for i in range(len(Results)):
        left=np.sum(Final_Results[i][Results[i]][85:115,70:92])
        right=np.sum(Final_Results[i][Results[i]][85:115,100:122])
        if left == 0:
            width_l=height_l = 0
        if right ==0:
            width_r=height_r = 0
        else:
            properties_left=regionprops(Final_Results[i][Results[i]][85:115,70:92][:,:,0])
            properties_right=regionprops(Final_Results[i][Results[i]][85:115,100:122][:,:,0])
            for p in properties_left:
                min_row_l, min_col_l, max_row_l, max_col_l = p.bbox
                width_l = max_col_l-min_col_l    
                height_l=max_row_l-min_row_l
                for p in properties_right:
                    min_row_r, min_col_r, max_row_r, max_col_r = p.bbox  
                width_r = max_col_r-min_col_r    
                height_r=max_row_r-min_row_r 
        Volume=0
        for j in range(Results[i]-7,Results[i]+7):
            Volume1 = np.sum(Final_Results[i][j])
            Volume=Volume+Volume1
        if i <130:    
            areas.append(["Baby"+str(i+214),"PLIC_l="+str(right),"PLIC_r="+str(left),"total="+str(left+right),
                      "LAT_l="+str(width_r),"AP_l="+str(height_r),"LAT_r="+str(width_l),"AP_r="+
                      str(height_l), "Vol="+str(Volume)])
            areas_short.append(["Baby"+str(i+214),right,left,left+right,width_r,height_r,width_l,height_l,Volume,len(Final_Results[i])])
        if i >=130:    
            areas.append(["Baby"+str(i+215),"PLIC_l="+str(right),"PLIC_r="+str(left),"total="+str(left+right),
                      "LAT_l="+str(width_r),"AP_l="+str(height_r),"LAT_r="+str(width_l),"AP_r="+
                      str(height_l), "Vol="+str(Volume)])
            areas_short.append(["Baby"+str(i+215),right,left,left+right,width_r,height_r,width_l,height_l,Volume,len(Final_Results[i])])  
    return areas,areas_short




def area_leftright_total_diam3(Final_Results,Results):
    areas=[]
    areas_short=[]
    for i in range(len(Results)):
        left=np.sum(Final_Results[i][Results[i]][85:115,70:92])
        right=np.sum(Final_Results[i][Results[i]][85:115,100:122])
        if left == 0:
            width_l=height_l = 0
        if right ==0:
            width_r=height_r = 0
        else:
            properties_left=regionprops(Final_Results[i][Results[i]][85:115,70:92][:,:,0])
            properties_right=regionprops(Final_Results[i][Results[i]][85:115,100:122][:,:,0])
            for p in properties_left:
                min_row_l, min_col_l, max_row_l, max_col_l = p.bbox
                width_l = max_col_l-min_col_l    
                height_l=max_row_l-min_row_l
                for p in properties_right:
                    min_row_r, min_col_r, max_row_r, max_col_r = p.bbox  
                width_r = max_col_r-min_col_r    
                height_r=max_row_r-min_row_r 
        Volume=0
        for j in range(Results[i]-7,Results[i]+7):
            Volume1 = np.sum(Final_Results[i][j])
            Volume=Volume+Volume1
        if i <1:    
            areas.append(["Baby"+str(i+389),"PLIC_l="+str(right),"PLIC_r="+str(left),"total="+str(left+right),
                      "LAT_l="+str(width_r),"AP_l="+str(height_r),"LAT_r="+str(width_l),"AP_r="+
                      str(height_l), "Vol="+str(Volume)])
            areas_short.append(["Baby"+str(i+389),right,left,left+right,width_r,height_r,width_l,height_l,Volume,len(Final_Results[i])])
        if i >=1 and i<55:    
            areas.append(["Baby"+str(i+390),"PLIC_l="+str(right),"PLIC_r="+str(left),"total="+str(left+right),
                      "LAT_l="+str(width_r),"AP_l="+str(height_r),"LAT_r="+str(width_l),"AP_r="+
                      str(height_l), "Vol="+str(Volume)])
            areas_short.append(["Baby"+str(i+390),right,left,left+right,width_r,height_r,width_l,height_l,Volume,len(Final_Results[i])]) 
        if i >=55 and i < 133:    
            areas.append(["Baby"+str(i+391),"PLIC_l="+str(right),"PLIC_r="+str(left),"total="+str(left+right),
                      "LAT_l="+str(width_r),"AP_l="+str(height_r),"LAT_r="+str(width_l),"AP_r="+
                      str(height_l), "Vol="+str(Volume)])
            areas_short.append(["Baby"+str(i+391),right,left,left+right,width_r,height_r,width_l,height_l,Volume,len(Final_Results[i])])              
        if i >=133:    
            areas.append(["Baby"+str(i+392),"PLIC_l="+str(right),"PLIC_r="+str(left),"total="+str(left+right),
                      "LAT_l="+str(width_r),"AP_l="+str(height_r),"LAT_r="+str(width_l),"AP_r="+
                      str(height_l), "Vol="+str(Volume)])
            areas_short.append(["Baby"+str(i+392),right,left,left+right,width_r,height_r,width_l,height_l,Volume,len(Final_Results[i])])      
    return areas,areas_short





def area_leftright_total_diam4(Final_Results,Results):
    areas=[]
    areas_short=[]
    for i in range(len(Results)):
        left=np.sum(Final_Results[i][Results[i]][85:115,70:92])
        right=np.sum(Final_Results[i][Results[i]][85:115,100:122])     
        if left == 0:
            width_l=height_l = 0
        if right ==0:
            width_r=height_r = 0
        else:
            properties_left=regionprops(Final_Results[i][Results[i]][85:115,70:92][:,:])
            properties_right=regionprops(Final_Results[i][Results[i]][85:115,100:122][:,:])
            for p in properties_left:
                min_row_l, min_col_l, max_row_l, max_col_l = p.bbox
                width_l = max_col_l-min_col_l    
                height_l=max_row_l-min_row_l
                for p in properties_right:
                    min_row_r, min_col_r, max_row_r, max_col_r = p.bbox  
                width_r = max_col_r-min_col_r    
                height_r=max_row_r-min_row_r 
        Volume=0
        for j in range(Results[i]-7,Results[i]+7):
            Volume1 = np.sum(Final_Results[i][j])
            Volume=Volume+Volume1
        if i <69:    
            areas.append(["Baby"+str(i+389),"PLIC_l="+str(right),"PLIC_r="+str(left),"total="+str(left+right),
                      "LAT_l="+str(width_r),"AP_l="+str(height_r),"LAT_r="+str(width_l),"AP_r="+
                      str(height_l), "Vol="+str(Volume)])
            areas_short.append(["Baby"+str(i+1),right,left,left+right,width_r,height_r,width_l,height_l,Volume,len(Final_Results[i])])
        if i >=69 :    
            areas.append(["Baby"+str(i+390),"PLIC_l="+str(right),"PLIC_r="+str(left),"total="+str(left+right),
                      "LAT_l="+str(width_r),"AP_l="+str(height_r),"LAT_r="+str(width_l),"AP_r="+
                      str(height_l), "Vol="+str(Volume)])
            areas_short.append(["Baby"+str(i+288),right,left,left+right,width_r,height_r,width_l,height_l,Volume,len(Final_Results[i])]) 

    return areas,areas_short

'''unet for axial training'''
def get_unet2():
    images = Input(shape=(64,64,1), name='images')

    conv1 = Conv2D(32, 3, 1, activation='relu', padding="same")(images)
    conv1 = Conv2D(32, 3, 1, activation='relu',   padding="same")(conv1)
    #print(conv1.shape)
    pool1 = MaxPool2D(pool_size=(2, 2), padding = "same")(conv1)

    conv2 = Conv2D(64, 3, 1, activation='relu',padding="same")(pool1)
    conv2 = Conv2D(64, 3, 1, activation='relu', padding='same')(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2),padding="same")(conv2)

    conv3 = Conv2D(128, 3, 1, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, 3, 1, activation='relu', padding='same')(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, 1, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, 3, 1, activation='relu', padding='same')(conv4)

    
    up7 = concatenate([Conv2DTranspose(128,kernel_size=(2,2), strides = (2,2),
                                                       padding="same", activation = "relu")(conv4), conv3], axis=-1)
    conv7 = Dropout(0.4)(up7)
   # print(up7.shape)
    conv7 = Conv2D(128, 3,1, activation='relu', padding='same')(up7)
    conv7 = Dropout(0.4)(conv7)
    conv7 = Conv2D(128, 3, 1, activation='relu', padding='same')(conv7)
   # print(conv7.shape)
    #print(conv2.shape)
    up8 = concatenate([Conv2DTranspose(64,kernel_size=(2,2), strides = (2,2),
                                                      padding='same',activation = "relu")(conv7), conv2], axis=-1)
    conv8 = Dropout(0.4)(up8)
    conv8 = Conv2D(64, 3, 1, activation='relu', padding='same')(up8)
    conv8 = Dropout(0.4)(conv8)
    conv8 = Conv2D(64, 3, 1, activation='relu', padding='same')(conv8)
    print(conv8.shape)
   # print(conv1.shape)

    up9 = concatenate([Conv2DTranspose(32,kernel_size=(2,2), strides = (2,2),
                                                       padding='same',activation = "relu")(conv8), conv1], axis=-1)
    conv9 = Dropout(0.4)(up9)
    conv9.shape
    conv9 = Conv2D(32, 3, 1, activation='relu', padding='same')(up9)
    conv9 = Dropout(0.4)(conv9)
    conv9 = Conv2D(32, 3, 1, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, 3, 1, activation = 'sigmoid', padding='same')(conv9)
    print(conv10.shape)
    out = conv10
    
    model = Model(inputs=images, outputs=out)
    
    model.compile(optimizer=SGD(lr=1e-3, momentum=0.99, decay=1e-2), loss=dice_loss, metrics=['accuracy'])

    return model
model2=get_unet2()

'''unet for sagittal and coronal training'''

def get_unet4():
    images = Input(shape=(56,56,1), name='images')

    conv1 = Conv2D(32, 3, 1, activation='relu', padding="same")(images)
    conv1 = Conv2D(32, 3, 1, activation='relu',   padding="same")(conv1)
    #print(conv1.shape)
    pool1 = MaxPool2D(pool_size=(2, 2), padding = "same")(conv1)

    conv2 = Conv2D(64, 3, 1, activation='relu',padding="same")(pool1)
    conv2 = Conv2D(64, 3, 1, activation='relu', padding='same')(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2),padding="same")(conv2)

    conv3 = Conv2D(128, 3, 1, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, 3, 1, activation='relu', padding='same')(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, 1, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, 3, 1, activation='relu', padding='same')(conv4)

    
    up7 = concatenate([Conv2DTranspose(128,kernel_size=(2,2), strides = (2,2),
                                                       padding="same", activation = "relu")(conv4), conv3], axis=-1)
    conv7 = Dropout(0.4)(up7)
   # print(up7.shape)
    conv7 = Conv2D(128, 3,1, activation='relu', padding='same')(up7)
    conv7 = Dropout(0.4)(conv7)
    conv7 = Conv2D(128, 3, 1, activation='relu', padding='same')(conv7)
   # print(conv7.shape)
    #print(conv2.shape)
    up8 = concatenate([Conv2DTranspose(64,kernel_size=(2,2), strides = (2,2),
                                                      padding='same',activation = "relu")(conv7), conv2], axis=-1)
    conv8 = Dropout(0.4)(up8)
    conv8 = Conv2D(64, 3, 1, activation='relu', padding='same')(up8)
    conv8 = Dropout(0.4)(conv8)
    conv8 = Conv2D(64, 3, 1, activation='relu', padding='same')(conv8)
    print(conv8.shape)
   # print(conv1.shape)

    up9 = concatenate([Conv2DTranspose(32,kernel_size=(2,2), strides = (2,2),
                                                       padding='same',activation = "relu")(conv8), conv1], axis=-1)
    conv9 = Dropout(0.4)(up9)
    conv9.shape
    conv9 = Conv2D(32, 3, 1, activation='relu', padding='same')(up9)
    conv9 = Dropout(0.4)(conv9)
    conv9 = Conv2D(32, 3, 1, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, 3, 1, activation = 'sigmoid', padding='same')(conv9)
    print(conv10.shape)
    out = conv10
    
    model = Model(inputs=images, outputs=out)
    
    model.compile(optimizer=SGD(lr=1e-3, momentum=0.99, decay=1e-3), loss=dice_loss, metrics=['accuracy'])

    return model
model4=get_unet4()
  

  

'''crossvalidation functions'''''''''''''''''''''''''''
'''creates cross validation data, i.e. generates N folds from data'''   
def cross_val(data,N, indices):
    maximum = len(data)
    factor = np.round(int(maximum)/N)
    folds=[]
    Patient=[]
  #  folds.append(data[:np.int(factor)])
    for i in range(0,N):
        fold = data[i*int(factor):(i+1)*int(factor)]
        P = [i*int(factor),(i+1)*int(factor)]
        folds.append(fold)
        Patient.append(P)
    final_folds=[]    
    for i in range(0,len(folds)): 
        fold = np.concatenate(np.asarray(folds[i]),axis=0)
        fold = fold[:,65:129,65:129]
        fold = np.expand_dims(fold,axis=-1)
        if np.max(fold)==1:
            f = np.abs(1-fold)
            fold1 = np.concatenate((fold,f), axis=-1)
            final_folds.append(fold1)    
        else:   
            Xnorm=[]
            for j in range(0,len(fold)):
                X_n =(fold[j]-np.mean(fold[j]))/np.std(fold[j])
                Xnorm.append(X_n)
            fold= np.asarray(Xnorm)
            final_folds.append(fold)
    Sets=[]
    for i in range(0,len(final_folds)):
        val = final_folds[i]
        training = np.concatenate([final_folds[j] for j in list([0,1,2,3,4]) if j!=i])
        Sets.append([val,training,Patient])
    return Sets


'''generates labels for classification network'''
def fold_slice_labels(Y_folds):
    new_list1=[]
    new_list2=[]
    for i in range(0,len(Y_folds)):
        M_val = Y_folds[i][0]#for val data
        M = np.sum(np.sum(M_val,axis = 2),axis=1)
        v1 = np.float32((M[:,0]>0))
        lab_val = np.stack((v1,1-v1),axis=-1)
        M_val = Y_folds[i][1]#for training data
        M = np.sum(np.sum(M_val,axis = 2),axis=1)
        v1 = np.float32((M[:,0]>0))
        lab_train = np.stack((v1,1-v1),axis=-1)
        new_list1.append(lab_val)
        new_list2.append(lab_train)
        liste=[new_list1,new_list2]
        liste = np.asarray(liste)
        liste = np.moveaxis(liste,0,1)
    Y_class=liste    
    return Y_class



#%%generate folds of coronal slices for training on coronal slices
def generate_folds_coronal(data):
    liste=[]
    for i in range(0,5):
        c = data[i*56*20:(i+1)*56*20]
        liste.append(c)
        Sets=[]    
    for i in range(0,len(liste)):
        val = liste[i]
        training = np.concatenate([liste[j] for j in list([0,1,2,3,4]) if j!=i])
        Sets.append(training)
    return([Sets,liste])      

#%%generate folds of coronal slices for training on sagittal slices
def generate_folds_sagittal(data):
    liste=[]
    for i in range(0,5):
        c = data[i*56*20:(i+1)*56*20]
        liste.append(c)
        Sets=[]    
    for i in range(0,len(liste)):
        val = liste[i]
        training = np.concatenate([liste[j] for j in list([0,1,2,3,4]) if j!=i])
        Sets.append(training)
    return([Sets,liste])   




def reconstruct_axial_crossval(data,ind):
    os.chdir("C:\\Users\\nadja\\Documents\\PLIC_programm\\Data_npz")
    f = np.load("Training_Babies.npz")
    X = f["T1"]
    indices = f["indices"]
    Matrix = np.zeros_like(X,dtype="float64")
    for i in range(0,5):
        ind1 = np.sum(indices[:20*i])
        Matrix[ind1+ind[i],65:129,65:129]=data[i]
    return Matrix  

def reconstruct_sag_crossval(X, levels):
    os.chdir("C:\\Users\\nadja\\Documents\\PLIC_programm\\Data_npz")
    f = np.load("Training_Babies.npz")
    X_org = f["T1"]
    indices = f["indices"]
    X_org = np.zeros_like(X_org)
    liste = []
#liste.append(T1[:indices[0]])
    for i in range(len(indices)):
        index1 = int(np.sum(indices[:i]))
        index2 = int(index1 + indices[i])
        P = X_org[index1:index2]
        P=np.asarray(P)
        liste.append(P)
    liste = np.asarray(liste)   
    Matrix=[]
    for i in range(len(liste)):
        M = np.zeros_like(np.moveaxis(liste[i],2,0), dtype="float64")
        Matrix.append(M)
   # Matrix = np.asarray(Matrix)    
    for j in range(len(levels)):
        index1 = int(j*56)
        index2 = int(j*56 +56)
        Matrix[j][68:124,int(levels[j])-28:int(levels[j])+28,68:124]=X[index1:index2,:,:]
        Matrix[j] = np.moveaxis(Matrix[j],0,2)
    Matrix=np.concatenate(Matrix,axis=0)    
    return Matrix



def reconstruct_cor_crossval(X, levels):
    os.chdir("C:\\Users\\nadja\\Documents\\PLIC_programm\\Data_npz")
    f = np.load("Training_Babies.npz")
    X_org = f["T1"]
    indices = f["indices"]
    X_org = np.zeros_like(X_org)
    liste = []
#liste.append(T1[:indices[0]])
    for i in range(len(indices)):
        index1 = int(np.sum(indices[:i]))
        index2 = int(index1 + indices[i])
        P = X_org[index1:index2]
        P=np.asarray(P)
        liste.append(P)
    liste = np.asarray(liste)   
    Matrix=[]
    for i in range(len(liste)):
        M = np.zeros_like(np.moveaxis(liste[i],0,1),dtype="float64")
        Matrix.append(M)
   # Matrix = np.asarray(Matrix)    
    for j in range(len(levels)):
        index1 = int(j*56)
        index2 = int(j*56 +56)
        Matrix[j][68:124,int(levels[j])-28:int(levels[j])+28,68:124]=X[index1:index2,:,:]
        Matrix[j] = np.moveaxis(Matrix[j],1,0)
    Matrix=np.concatenate(Matrix,axis=0)    
    return Matrix

'''---------------------------------------'''''''''''''''''''''''''''''''''''''''''''''''''''

'''----------------------------evaluation metrics------------------------------------------------'''
'''dice coeff'''
def evaluate_dice(X,Y):
    dices=[]
    dices_all=[]
    intersection = np.logical_and(X,Y)
    union = np.logical_or(X,Y)
    iou_score = np.sum(intersection) / np.sum(union)
    intersection = np.sum(X[Y==1]) * 2.0
    dice = intersection / (np.sum(X) + np.sum(Y))
    print(iou_score)
    # Dice similarity function
    for j in range(0,len(X)):
        intersection = np.sum(X[j][Y[j]==1]) * 2.0
        if np.sum(Y[j])>2:
            dice_per_patient=intersection/(np.sum(X[j]) + np.sum(Y[j])+0.00001)
            dices_all.append(dice_per_patient)
    dices.append(dice)
    return dices_all,dices


def find_best_dice(combi,Labels):
    dices=[]
    Liste = [0.3333333,0.66666,1,1.25,1.33,1.5,1.666,2,2.333333,2.6666666]
    for i in Liste:
        m = np.copy(combi)
        m[m<i]=0
        m[m>0]=1
        dice=evaluate_dice(m,Labels)[0]
        dices.append(dice)
    return dices


import numpy as numpy
import numpy
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion,\
    generate_binary_structure
from scipy.ndimage.measurements import label, find_objects
from scipy.stats import pearsonr
def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = numpy.asarray(voxelspacing, dtype=numpy.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
            
    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)
    
    # test for emptiness
    if 0 == numpy.count_nonzero(result): 
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == numpy.count_nonzero(reference): 
        raise RuntimeError('The second supplied array does not contain any binary object.')    
            
    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    
    # compute average surface distance        
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    
    return sds


def positive_predictive_value(result, reference):
    """
    Positive predictive value.
    Same as :func:`precision`, see there for a detailed description.
    
    See also
    --------
    :func:`true_positive_rate`
    :func:`true_negative_rate`
    """
    return precision(result, reference)

def jc(result, reference):
    """
    Jaccard coefficient
    
    Computes the Jaccard coefficient between the binary objects in two images.
    
    Parameters
    ----------
    result: array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.
    reference: array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.
    Returns
    -------
    jc: float
        The Jaccard coefficient between the object(s) in `result` and the
        object(s) in `reference`. It ranges from 0 (no overlap) to 1 (perfect overlap).
    
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))
    
    intersection = numpy.count_nonzero(result & reference)
    union = numpy.count_nonzero(result | reference)
    
    jc = float(intersection) / float(union)
    
    return jc
#https://github.com/loli/medpy/blob/master/medpy/metric/binary.py
def hd95(result, reference, voxelspacing=None, connectivity=1):
    """
    95th percentile of the Hausdorff Distance.
    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.
    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of 
        elements along each dimension, which is usually given in mm.
    See also
    --------
    :func:`hd`
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    return hd95

def evaluate_jc(X,Y):
    #hds=[]
    hd_all=[]
    #hd = hd95(X,Y)
    # Dice similarity function
    for j in range(0,len(X)):
        if (np.sum(X[j])>2 and np.sum(Y[j])>2):
            hd_per_patient=jc(X[j],Y[j])
            hd_all.append(hd_per_patient)
    #hds.append(hd)
    return hd_all

def find_best_jc(combi,Labels):
    dices=[]
    Liste = [0.3333333,0.66666,1,1.33,1.666,2,2.333333,2.6666666]
    for i in Liste:
        m = np.copy(combi)
        m[m<i]=0
        m[m>0]=1
        dice=evaluate_jc(m,Labels)
        dice = np.asarray(dice)
        dices.append(dice)
    return dices



def evaluate_hd(X,Y):
    #hds=[]
    hd_all=[]
    #hd = hd95(X,Y)
    # Dice similarity function
    for j in range(0,len(X)):
        if (np.sum(X[j])>2 and np.sum(Y[j])>2):
            hd_per_patient=hd95(X[j],Y[j])
            hd_all.append(hd_per_patient)
    #hds.append(hd)
    return hd_all

def find_best_hd(combi,Labels):
    dices=[]
    Liste = [0.3333333,0.66666,1,1.33,1.666,2,2.333333,2.6666666]
    for i in Liste:
        m = np.copy(combi)
        m[m<i]=0
        m[m>0]=1
        dice=evaluate_hd(m,Labels)
        dice = np.asarray(dice)
        dices.append(dice)
    return dices


def evaluate_prec(X,Y):
    #hds=[]
    hd_all=[]
    #hd = hd95(X,Y)
    for j in range(0,len(X)):
        if (np.sum(X[j])>2 and np.sum(Y[j])>2):
            hd_per_patient=precision(X[j],Y[j])
            hd_all.append(hd_per_patient)
    #hds.append(hd)
    return hd_all

from sklearn.metrics import recall_score
def evaluate_recall(X,Y):
    #hds=[]
    hd_all=[]
    #hd = hd95(X,Y)
    # Dice similarity function
    for j in range(0,len(X)):
        if (np.sum(X[j])>2 and np.sum(Y[j])>2):
            hd_per_patient=recall_score(Y[j,65:129,65:129].flatten(),X[j,65:129,65:129].flatten())
            hd_all.append(hd_per_patient)
    #hds.append(hd)
    return hd_all


def precision(X,Y):
    return average_precision_score(Y[65:129,65:129].flatten(),X[65:129,65:129].flatten())

def recall(X,Y):
    return recall_score(Y[65:129,65:129].flatten(),X[65:129,65:129].flatten())




from sklearn.metrics import average_precision_score
def evaluate_precision(X,Y):
    #hds=[]
    hd_all=[]
    #hd = hd95(X,Y)
    # Dice similarity function
    for j in range(0,len(X)):
        if (np.sum(X[j])>2 and np.sum(Y[j])>2):
            hd_per_patient=average_precision_score(Y[j,65:129,65:129].flatten(),X[j,65:129,65:129].flatten())
            hd_all.append(hd_per_patient)
    #hds.append(hd)
    return hd_all


def find_best_precision(combi,Labels):
    dices=[]
    Liste = [0.3333333,0.66666,1,1.33,1.666,2,2.333333,2.6666666]
    for i in Liste:
        m = np.copy(combi)
        m[m<i]=0
        m[m>0]=1
        dice=evaluate_precision(m,Labels)
        dices.append(dice)
    return dices

def find_best_recall(combi,Labels):
    dices=[]
    Liste = [0.3333333,0.66666,1,1.33,1.666,2,2.333333,2.6666666]
    for i in Liste:
        m = np.copy(combi)
        m[m<i]=0
        m[m>0]=1
        dice=evaluate_recall(m,Labels)
        dices.append(dice)
    return dices
