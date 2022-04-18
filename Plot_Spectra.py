'''
GUI 构建
'''
import numpy as np
import os,sys,csv,random
from matplotlib import pyplot as plt
from math import *

font1 = {'family': 'Times New Roman','weight' : 'normal','size': 30,}
fontnumsize = 24
plt.rc('font', size=16)#size of fig axis
path =  os.path.dirname(sys.argv[0])

Fig_Path = os.path.join(path,'Fig_Path')
if not os.path.exists(Fig_Path): 
    os.mkdir(Fig_Path)

#===lines location files
cnn_high_line_csv= './lines/high_snr_line.csv'
cnn_mid_line_csv= './lines/mid_snr_line.csv'
emi_line_csv= './lines/galaxylines_EL.csv'
abs_line_csv= './lines/galaxylines_ABS.csv'

#===cnn line
cnn_high_line = []
with open(cnn_high_line_csv) as f:
    reader=csv.DictReader(f)
    for row in reader:
        temp_key = {'Name':row['lines_name'],'Lamda0':float(row['lines'])}
        cnn_high_line.append(temp_key)
cnn_mid_line = []
with open(cnn_mid_line_csv) as f:
    reader=csv.DictReader(f)
    for row in reader:
        temp_key = {'Name':row['lines_name'],'Lamda0':float(row['lines'])}
        cnn_mid_line.append(temp_key)
#===emision lines
emi_line = []
with open(emi_line_csv) as f:
    reader=csv.DictReader(f)
    for row in reader:
        temp_key = {'Name':row['lines_name'],'Lamda0':float(row['lines'])}
        emi_line.append(temp_key)
#===absorption lines
abs_line = []
with open(abs_line_csv) as f:
    reader=csv.DictReader(f)
    for row in reader:
        temp_key = {'Name':row['lines_name'],'Lamda0':float(row['lines'])}
        abs_line.append(temp_key)

#=====plot the raw spectra
def Plot_Raw_Spectra(gal_info):
    loglam0,pixel,loglam_step =  gal_info['loglam0'],gal_info['pixel'],gal_info['loglam_step']
    lamda=10**(loglam0+np.arange(pixel)*loglam_step)
    flux = gal_info['flux']
    ZG,SNR,RA,DEC, = gal_info['ZG'],round(gal_info['sn_mean'],3),round(gal_info['RA'],6),round(gal_info['DEC'],6)
    ZE = 0
    ID = str(gal_info['PLATE'])+'-'+str(gal_info['MJD'])+'-'+str(gal_info['FIBER'])

    plt.figure(figsize=(30,6), dpi=360)
    #===plot flux
    ax1 = plt.subplot(111)
    ax1.plot(lamda, flux,'-k',linewidth=0.1,label="raw_spec")
    ax1.legend(loc='upper right',prop=font1) 
    
    #====cnn lines / high snr
    for line in cnn_high_line:
        x0 = line['Lamda0']*(1+ZG)
        if x0<3700 or x0>10000:
            continue
        plt.axvline(x0,lw=0.5,linestyle='-.',c='b',alpha=1)
        ax1.text(x0-20, 10, line['Name'],fontsize=6,color='b',alpha=0.5)
    #====cnn lines / mid snr
    for line in cnn_mid_line:
        x0 = line['Lamda0']*(1+ZG)
        if x0<3700 or x0>10000:
            continue
        plt.axvline(x0,lw=0.5,linestyle='-.',c='b',alpha=1)
        ax1.text(x0-20, 10, line['Name'],fontsize=6,color='b',alpha=0.5)
    ZG = round(ZG,3)
    #=== plot the ZE emission lines
    if 'ZE' in gal_info.keys() and gal_info['ZE']!=0:
        ZE = gal_info['ZE']
        #====cnn lines / high snr
        for line in cnn_high_line:
            x0 = line['Lamda0']*(1+ZE)
            if x0<3700 or x0>10000:
                continue
            plt.axvline(x0,lw=0.5,linestyle='-.',c='r',alpha=1)
            ax1.text(x0-20, 10, line['Name'],fontsize=6,color='r',alpha=0.5)
        #====cnn lines / mid snr
        for line in cnn_mid_line:
            x0 = line['Lamda0']*(1+ZE)
            if x0<3700 or x0>10000:
                continue
            plt.axvline(x0,lw=0.5,linestyle='-.',c='r',alpha=1)
            ax1.text(x0-20, 10, line['Name'],fontsize=6,color='r',alpha=0.5)
        ZE = round(ZE,3)
        
    #==the title and axis unit
    plt.tick_params(labelsize=fontnumsize)
    plt.ylabel('flux (10$^{-17}$ erg/s/cm$^{2}$/Ang)',font1)
    plt.xlabel('wavelength ($/AA$)',font1)
    plt.title('ID:'+ID+' ZG:'+str(ZG)+' ZE:'+str(ZE)+' SNR:'+str(SNR)+' RA:'+str(RA)+' DEC:'+str(DEC))
    plt.tight_layout()
    #===save, name: PLATE-MJD-FIBER
    fname= os.path.join(Fig_Path,ID+'.pdf')
    plt.savefig(fname)
    plt.close()
    print('ploting fig ..., save in',fname)

