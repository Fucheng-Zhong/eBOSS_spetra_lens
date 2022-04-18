#! -*- coding: utf-8 -*-
'''
predicted the value of PL, ZE & ZG
and save the PL>0.95 candidates
'''
from keras.models import Model
import numpy as np
import pandas as pd
import cmodel,os
Predict_Result = 'Predict_Result'
if not os.path.exists(Predict_Result):
    os.mkdir(Predict_Result)

#=====保存候选者
def Save_candida(SPE_INFO,fname):
    fname = fname + '-num='+str(len(SPE_INFO))
    fname = os.path.join(Predict_Result,fname)
    np.save(fname,SPE_INFO,allow_pickle=True,fix_imports=True)

#========input of CNNs
def Read_data(info):
     x = []
     for one in info:
          x.append(one['flux'])
     X = np.array(x)
     X = X.reshape(X.shape+(1,))
     return X

#====print the predicted value to csv files
def SaveResultcsv(info,fname):
    fname = fname.split('/')[-1].split('.')[-2]
    fname = os.path.join(Predict_Result,fname+'-num='+str(len(info)))+'.csv'#===name of csv files
    saveinfo,temp = [],{}
    for oneinfo in info:
        for indx in oneinfo.keys():
            temp[indx] = oneinfo[indx] 
        saveinfo.append(temp.copy())
    df = pd.DataFrame(saveinfo)
    df.to_csv(fname, mode="w", header=True, index=True)#==keep the row index
        
#========predicted by 3 CNNs, P is the threshold of PL
def Predic_CNN(spec_file,P=0.95,sav=0): 
    SPE_INFO = np.load(spec_file,allow_pickle=True,fix_imports=True)
    X = Read_data(SPE_INFO)
    
    model_PL = Model(inputs=cmodel.Inpt, outputs=cmodel.OutPut_PL)
    model_PL.load_weights(cmodel.modelfile+'weights_PL.h5')
    Pred_PL = model_PL.predict(X)#give PL

    model_ZE = Model(inputs=cmodel.Inpt, outputs=cmodel.OutPut_ZE)
    model_ZE.load_weights(cmodel.modelfile+'weights_ZE.h5')
    Pred_ZE = model_ZE.predict(X)#give ZE

    model_ZG = Model(inputs=cmodel.Inpt, outputs=cmodel.OutPut_ZG)
    model_ZG.load_weights(cmodel.modelfile+'weights_ZG.h5')
    Pred_ZG = model_ZG.predict(X)#give ZG
    
    #===predict result, turn to float format
    for i in range(len(SPE_INFO)):
        SPE_INFO[i]['PL'],SPE_INFO[i]['PZE'],SPE_INFO[i]['PZG'] = float(Pred_PL[i]),float(Pred_ZE[i]),float(Pred_ZG[i])
    #====print result to csv files
    SaveResultcsv(SPE_INFO,spec_file)
    fname = spec_file.split('/')[-1].split('.')[-2]
    if sav == 1: #====option, if 1, save the candidates' spectra
        save_info = []
        for one in SPE_INFO:
            if one['PL'] >= P:
                save_info.append(one)
        Save_candida(save_info,fname+'-P='+str(P))

if __name__ == '__main__':
    spec_file = './CNN_Data/val_data_labeled.npy'
    Predic_CNN(spec_file,P=0.95,sav=1)
    spec_file = './Raw_SDSS_Spectra/topre_data.npy'
    Predic_CNN(spec_file,P=0.95,sav=1)