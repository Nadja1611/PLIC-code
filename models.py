# -*- coding: utf-8 -*-
"""
Created on Fri May  6 07:25:05 2022

@author: Nadja
"""

import os
os.chdir("C://Users//nadja//Documents//PLIC_programm")
from functions import *

#%% build PLIC-Slice-Selector
def PLIC_Slice_Selector():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(64,64,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss="categorical_crossentropy",optimizer='Adam',metrics=['accuracy'])
    model_checkpoint=ModelCheckpoint('weights_classification.hdf5',save_best_only=True,monitor='val_loss')
    return model


#%% Buld Thalamic Slice Selector model
def Thalamic_Slice_Selector():
    model3 = Sequential()
    model3.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(64,64,1)))
    model3.add(Conv2D(64, (3, 3), activation='relu'))
    model3.add(MaxPool2D(pool_size=(2, 2)))
    model3.add(Conv2D(64, (3, 3), activation='relu'))
    model3.add(MaxPool2D(pool_size=(2, 2)))
    model3.add(Flatten())
    model3.add(Dense(64, activation='relu'))
    model3.add(Dense(16, activation='relu'))
    model3.add(Dense(1, activation='relu'))
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001, name='Adam')
#KL = tf.keras.losses.KLDivergence()
    model3.compile(optimizer=opt,loss="MSE",metrics=['MSE'])
    model_checkpoint=ModelCheckpoint('weights_thalamus_KL.hdf5',save_best_only=True,monitor='loss')
    return model3


#%% build axial segmentation module
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
    
    model.compile(optimizer=SGD(lr=1.5e-3, momentum=0.99, decay=1e-3), loss= generalized_dice_loss(weight_bg,weight_fg), metrics=['accuracy'])

    return model
model2=get_unet2()


#%% build sagittal and coronal segmentation module

def get_unet_sag_cor():
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
model4=get_unet_sag_cor()