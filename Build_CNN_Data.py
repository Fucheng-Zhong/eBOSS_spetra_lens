'''
It is used to add the labeled and add artificial emission lines
1. training data
2. testing data
3. validation data
'''
import numpy as np
import os,random,csv,SDSS_Spetra_Produce,Plot_Spectra
from math import *

#====setting the maximum ZE and minimal delta redshift between ZE & ZG
Max_ZE = 1.2
Delat_Z = 0.1

#==Generate folder
PATH = './CNN_Data'
if not os.path.exists(PATH): 
    os.mkdir(PATH)
    
#====del the no need parameter
del_keys = ['invar']

#===lines location files
cnn_high_line_csv= './lines/high_snr_line.csv'
cnn_mid_line_csv= './lines/mid_snr_line.csv'
#===loading cnn line
cnn_high_line = [] #==high snr
with open(cnn_high_line_csv) as f:
    reader=csv.DictReader(f)
    for row in reader:
        temp_key = {'Name':row['lines_name'],'Lamda0':float(row['lines']),'sigma1':float(row['sigma1']),'sigma2':float(row['sigma2']),'h1':float(row['h1']),'h2':float(row['h2'])}
        cnn_high_line.append(temp_key)
cnn_mid_line = [] #===mid snr
with open(cnn_mid_line_csv) as f:
    reader=csv.DictReader(f)
    for row in reader:
        temp_key = {'Name':row['lines_name'],'Lamda0':float(row['lines']),'sigma1':float(row['sigma1']),'sigma2':float(row['sigma2']),'h1':float(row['h1']),'h2':float(row['h2'])}
        cnn_mid_line.append(temp_key)

#====flux of lines, shift by z
def Emiline_Flux(gal,cnn_lines,z):
    loglam0,pixel,loglam_step =  gal['loglam0'],gal['pixel'],gal['loglam_step']
    lamda=10**(loglam0+np.arange(pixel)*loglam_step)
    emi_flux = [0]*len(lamda) #===null flux
    for oneline in cnn_lines:
        #===setting parameters
        low = (1+z)**2/2
        high = random.uniform(oneline['h1'],oneline['h2'])/low
        sigma1 = random.uniform(1*oneline['sigma1'],2*oneline['sigma1'])*(1+z)
        center_wave= oneline['Lamda0']*(1+z)
        if oneline['Name'] == '[OII]':
            flux = high*np.exp(-(lamda-center_wave)**2/(2*sigma1**2))
            emi_flux = emi_flux + flux
        else:#===the singlet line, one gaussian
            flux = high*np.exp(-(lamda-center_wave)**2/(2*sigma1**2))
            emi_flux = emi_flux + flux
    return emi_flux

#====Add emission line to background
def Add_Emiline_to_Background(gal_info,ZB):
    #==add high snr and mid snr lines
    add_lines = cnn_high_line + cnn_mid_line   
    emi_flux = Emiline_Flux(gal_info,add_lines,ZB)
    return emi_flux

#====Add emission line to Foreground
def Add_Emiline_to_Foreground(gal_info,ZG):
    #==only add high snr lines
    add_lines = cnn_high_line
    emi_flux = Emiline_Flux(gal_info,add_lines,ZG)
    return emi_flux

#====Simulate sample, from real spectrum
def Build_Train_Data(Raw_data):
    tra_info = []
    print('Now is reading data',Raw_data)
    gal_info = np.load(Raw_data,allow_pickle=True,fix_imports=True)
    count = 0
    for one_gal in gal_info:
        count = count + 1
        ZG = one_gal['ZG']
        ZE = np.random.uniform(ZG+Delat_Z,Max_ZE) #===randomly decide the ZE
        neg_one = one_gal.copy()
        #===delect the no need parameter, in order to reduce space
        for one_key in del_keys:
            if one_key in neg_one.keys():
                del neg_one[one_key]
        #===building negative sample
        neg_one['ZE'],neg_one['lable'] = 0,0 #===negative sample lable 0
        #===add forground emission lines(ZG)
        if count%5 == 0: 
            fore_emi = Add_Emiline_to_Foreground(neg_one,neg_one['ZG'])
            neg_one['flux'] = neg_one['flux'] + fore_emi
        tra_info.append(neg_one.copy())
        #====positive sample, add background emission lines(ZE)
        pos_one = neg_one.copy()
        pos_one['ZE'],pos_one['lable'] = ZE,1 #===positive sample lable 1
        back_emi = Add_Emiline_to_Background(pos_one,ZE)
        pos_one['flux'] = neg_one['flux'] + back_emi
        tra_info.append(pos_one.copy())   
    #=====saving in the npy form and print in csv flie
    name = Raw_data.split('/')[-1].split('.')[0]
    fname = os.path.join(PATH, name +'_labeled')
    np.save(fname,tra_info,allow_pickle=True,fix_imports=True)
    SDSS_Spetra_Produce.SaveResultcsv(tra_info,fname)
    print('Now is saving Data###',fname)

#===building the training, validation and testing samples
def Building_data():
    tempfile = os.path.join(SDSS_Spetra_Produce.PATH,'tra_data.npy')
    Build_Train_Data(tempfile)
    tempfile = os.path.join(SDSS_Spetra_Produce.PATH,'val_data.npy')
    Build_Train_Data(tempfile)
    tempfile = os.path.join(SDSS_Spetra_Produce.PATH,'test_data.npy')
    Build_Train_Data(tempfile)
    
if __name__=="__main__":
    Building_data()
    #=== plot the first 11 picture, for checking !
    galaxy_spetra_file = os.path.join(PATH,'val_data_labeled.npy')
    galaxy_spetra_info = np.load(galaxy_spetra_file,allow_pickle=True,fix_imports=True)
    i = 0
    for one in galaxy_spetra_info:
        Plot_Spectra.Plot_Raw_Spectra(one)
        i = i + 1
        if i > 10:
            break