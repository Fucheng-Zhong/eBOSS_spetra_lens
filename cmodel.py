#! -*- coding: utf-8 -*-
'''
Training the CNNs.
3 CNNs are be used here.
CNN1 is used to give the probability of lens, PL
CNN2 is used to give the redshift of background, ZE
CNN3 is used to give the redshift of foreground, ZG
'''
import numpy as np
import os,SDSS_Spetra_Produce
import keras
#from keras.callbacks import LearningRateScheduler,Callback
from keras.models import Model
from keras.layers import *
from keras import regularizers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger

modelfile = './model_CNN/'
if not os.path.exists(modelfile):
     os.makedirs(modelfile)

from keras import backend as K
#====huber_loss
def huber_loss(y_true, y_pred, delta=0.01):
    error = y_pred - y_true
    abs_error = K.abs(error)
    quadratic = K.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * K.square(quadratic) + delta * linear

learn_rate = [0.001,0.001,0.001] #==learn rate for 3 CNNs
epo = [1,1,1]   #==training epoch for 3 CNNs
batch = [32,32,32] #== batch for 3 CNNs
loss_fun = ['binary_crossentropy',huber_loss,huber_loss]#==loss function for 3 CNNs
drop_rate = [0,0,0]#=== drop rate for 3 CNNs
patien = [5,5,5] #====the patien for 3 CNNs, reach the patien the training will stop
mindelta = [1e-5,1e-5,1e-5] #===minimum fluctuation, below this will be deem as no improve.

print('loss fun:',loss_fun,'drop rate:',drop_rate,'patience:',patien,'min delta:',mindelta)

Inpt = Input(shape=(SDSS_Spetra_Produce.input_shape,1))#====the shape of spectra
def Model_stru(Drop_rate,actfun,l2=0):
     Kersize = 3
     #x=BatchNormalization(axis=1)(Inpt)#Normalize layer
     x=Conv1D(16, kernel_size=Kersize, activation='relu', padding='same')(Inpt)
     x= MaxPooling1D(pool_size=1,padding='same')(x)
     x=Conv1D(32, kernel_size=Kersize, activation='relu', padding='same')(x)
     x = AveragePooling1D(pool_size=2,padding='same')(x)
     x=Conv1D(64, kernel_size=Kersize, activation='relu', padding='same')(x)
     x = AveragePooling1D(pool_size=3,padding='same')(x)
     x=Conv1D(128, kernel_size=Kersize, activation='relu', padding='same')(x)
     x = AveragePooling1D(pool_size=3,padding='same')(x)
     x=Conv1D(256, kernel_size=Kersize, activation='relu', padding='same')(x)
     x = AveragePooling1D(pool_size=3,padding='same')(x)
     x=Conv1D(512, kernel_size=Kersize, activation='relu', padding='same')(x)
     x = AveragePooling1D(pool_size=3,padding='same')(x)
     x = Flatten()(x)
     x = Dropout(rate=Drop_rate)(x) #drop rate setting
     x = Dense(1024, activation='relu',kernel_regularizer=regularizers.l2(l2))(x)#set the l2 regularization
     x = Dropout(rate=Drop_rate)(x) #drop rate setting
     x = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(l2))(x)#set the l2 regularization
     if actfun == '':
          return Dense(1,)(x)
     else:
          return Dense(1,activation=actfun)(x)

OutPut_PL = Model_stru(Drop_rate=drop_rate[0],actfun='sigmoid',l2=0)
OutPut_ZE = Model_stru(Drop_rate=drop_rate[1],actfun='',l2=0)
OutPut_ZG = Model_stru(Drop_rate=drop_rate[2],actfun='',l2=0)

#====CNNs
def train_model(info_tra,info_val):
     #=============give PL
     if epo[0]>0:
          model_PL = Model(inputs=Inpt, outputs=OutPut_PL)
          model_PL.summary()
          model_PL.compile(optimizer=Adam(lr=learn_rate[0]),loss=loss_fun[0],metrics='acc')
          if os.path.exists(modelfile+'/weights_PL.h5'):
               model_PL.load_weights(modelfile+'/weights_PL.h5')
               print("CNN_PL_checkpoint_loaded")
          x_PL_tra,y_PL_tra = Label_data_PL(info_tra)#==label training data
          x_PL_val,y_PL_val = Label_data_PL(info_val)#==label validation data
          #===save every one model during training
          #filepath = modelfile+"/weights_{epoch:03d}-{val_acc:.4f}.h5"
          #checkpoint_PL = ModelCheckpoint(filepath, monitor='val_acc',verbose=1,save_best_only=False,period=1)
          #===only save the best model
          checkpoint_PL = ModelCheckpoint(modelfile+'/weights_PL.h5', monitor='val_acc',verbose=1,save_best_only=True)
          callback_PL = keras.callbacks.EarlyStopping(monitor='val_acc',mode='max', patience=patien[0],min_delta=mindelta[0])
          csv_logger_PL = CSVLogger(modelfile+"/model_PL_history.csv", append=True)#===save training history
          history_PL = model_PL.fit(x_PL_tra,y_PL_tra,epochs=epo[0],batch_size=batch[0], validation_data=(x_PL_val,y_PL_val),callbacks=[callback_PL,checkpoint_PL,csv_logger_PL],shuffle=True)
     #=============give ZE 
     if epo[1]>0:
          model_ZE = Model(inputs=Inpt, outputs=OutPut_ZE)
          model_ZE.summary()
          model_ZE.compile(optimizer=Adam(lr=learn_rate[1]),loss=loss_fun[1],metrics='mae')
          if os.path.exists(modelfile+'/weights_ZE.h5'):
               model_ZE.load_weights(modelfile+'/weights_ZE.h5')
               print("CNN_ZE_checkpoint_loaded")
          x_ZE_tra,y_ZE_tra = Label_data_ZE(info_tra)#==label training data
          x_ZE_val,y_ZE_val = Label_data_ZE(info_val)#==label validation data
          checkpoint_ZE = ModelCheckpoint(modelfile+'/weights_ZE.h5', monitor='val_mae',verbose=1, save_best_only=True)
          callback_ZE = keras.callbacks.EarlyStopping(monitor='val_mae',mode='min', patience=patien[1],min_delta=mindelta[1])
          csv_logger_ZE = CSVLogger(modelfile+"/model_ZE_history.csv", append=True)#===save training history
          history_ZE = model_ZE.fit(x_ZE_tra,y_ZE_tra,epochs=epo[1],batch_size=batch[1], validation_data=(x_ZE_val,y_ZE_val),callbacks=[callback_ZE,checkpoint_ZE,csv_logger_ZE],shuffle=True)
     #=============give ZG
     if epo[2]>0:
          model_ZG = Model(inputs=Inpt, outputs=OutPut_ZG)
          model_ZG.summary()
          model_ZG.compile(optimizer=Adam(lr=learn_rate[2]),loss=loss_fun[2],metrics='mae')
          if os.path.exists(modelfile+'/weights_ZG.h5'):
               model_ZG.load_weights(modelfile+'/weights_ZG.h5')
               print("CNN_ZG_checkpoint_loaded")
          x_ZG_tra,y_ZG_tra = Label_data_ZG(info_tra)#==label training data
          x_ZG_val,y_ZG_val = Label_data_ZG(info_val)#==label validation data
          #===using val_mae as monitor
          checkpoint_ZG = ModelCheckpoint(modelfile+'/weights_ZG.h5', monitor='val_mae',verbose=1, save_best_only=True)
          callback_ZG = keras.callbacks.EarlyStopping(monitor='val_mae',mode='min', patience=patien[2],min_delta=mindelta[2])
          csv_logger_ZG = CSVLogger(modelfile+"/model_ZG_history.csv", append=True)#===save training history
          history_ZG = model_ZG.fit(x_ZG_tra,y_ZG_tra,epochs=epo[2],batch_size=batch[2], validation_data=(x_ZG_val,y_ZG_val),callbacks=[callback_ZG,checkpoint_ZG,csv_logger_ZG],shuffle=True)

#========label for CNN1
def Label_data_PL(gal_info):
     print('reading PL data....')
     x,y = [] , []
     for one in gal_info:
          x.append(one['flux'])
          y.append(one['lable'])
     #====disrupt the order
     idxs = list(range(len(x)))
     np.random.shuffle(idxs)
     X,Y = np.array(x)[idxs], np.array(y)[idxs]
     return X,Y   
#========label for CNN2
def Label_data_ZE(gal_info):
     print('reading Z_line data....')
     x,y = [],[]
     for one in gal_info:
          if one['lable'] == 1:
               x.append(one['flux'])
               y.append(one['ZE'])
     #====disrupt the order
     idxs = list(range(len(x)))
     np.random.shuffle(idxs)
     X,Y = np.array(x)[idxs], np.array(y)[idxs]
     return X,Y     
#========label for CNN3
def Label_data_ZG(gal_info):
     print('reading ZG data....')
     x,y = [],[]
     for one in gal_info:
          x.append(one['flux'])
          y.append(one['ZG'])
     #====disrupt the order
     idxs = list(range(len(x)))
     np.random.shuffle(idxs)
     X,Y = np.array(x)[idxs], np.array(y)[idxs]
     return X,Y

def TrainFun(epoh,rate):
     global epo,learn_rate
     epo,learn_rate = epoh,rate #=== renew the epoch & learn rate for 3 CNNs
     #=====training and validation data
     file_tra = './CNN_Data/tra_data_labeled.npy'
     file_val = './CNN_Data/val_data_labeled.npy'
     print('loading train file:',file_tra,'\n loading val file:',file_val)
     info_tra = np.load(file_tra,allow_pickle=True,fix_imports=True)
     info_val = np.load(file_val,allow_pickle=True,fix_imports=True)
     train_model(info_tra,info_val)

if __name__ == '__main__':
     #===suggest learn rate begin 0.0001
     ######===if CNNs no convergence, try several times...
     #TrainFun([20,20,20],[1e-4,1e-4,1e-4])
     TrainFun([10,10,10],[1e-5,1e-4,1e-4])
     #TrainFun([20,20,20],[1e-6,1e-5,1e-5])