'''
1.  Bin the slected out spectra. Every 0.05 redshift
2.  Divide data into 4 pieces. train, validation, test, prediction samples
3.  Align the data 
'''
import numpy as np
import pandas as pd
import os,math,Plot_Spectra

#===traing data, testing data, validation data number
#===for every bin !!
tra_num,test_num,val_num = 2000,100,100

#===set the redshift bin setting
binnum = 15   #===redshift 0.05-0.8
Redshift_bin = (np.arange(binnum+1)+1)*0.05
#==Generate folder
PATH = './Raw_SDSS_Spectra'
if not os.path.exists(PATH): 
    os.mkdir(PATH)

#====cut the spectra into a fitting shape
lambda_min,lambda_max = 3700,9202
loglam_min,loglam_max = (math.log(lambda_min,10),math.log(lambda_max,10))
input_shape = int((loglam_max - loglam_min)/1e-4)
print('lambda:',lambda_min,lambda_max,'loglam:',loglam_min,loglam_max,'shape',input_shape)
    
#====Binning the samples, according redshift.
def Redshift_Bin(fname):
    gal_info = np.load(fname,allow_pickle=True,fix_imports=True)
    SaveResultcsv(gal_info,fname)#===print the value to csv file
    #====disrupt the order of spectra
    idxs = list(range(len(gal_info)))
    np.random.shuffle(idxs)
    gal_info = np.array(gal_info)[idxs]
    #====binning the samples, Redshift_bin are different bin range 
    z_bin = [[ ] for i in range(binnum)]
    print('redshift bin:',Redshift_bin)
    for one_gal in gal_info:#===bin the data
        for i in range(len(Redshift_bin)-1):
            if Redshift_bin[i]<=one_gal['ZG']<Redshift_bin[i+1]:
                one_gal = Cut_flux(one_gal)#===cut spetra in same pixel range
                if one_gal != 0:
                    z_bin[i].append(one_gal)
                    print(len(one_gal['flux']),one_gal['pixel'],one_gal['loglam0'])
                break
    #====show the number of spetra in each bin
    for i in range(len(z_bin)):
        print('the len of onebin:',len(z_bin[i])) 
    #====saving the binned spectra
    fname = os.path.join(PATH,'Bin_data')+'.npz'
    print('save the npz file######',fname)
    np.savez_compressed(fname,*z_bin)
    return fname

#====cut-out the spectrum to fit the cnn
def Cut_flux(one_gal):
    low_index = int((loglam_min- one_gal['loglam0'])/one_gal['loglam_step'])
    up_index =  low_index + input_shape
    flux,invar = one_gal['flux'].copy(),one_gal['invar'].copy()
    if up_index > len(one_gal['flux']) or low_index < 0: #make sure that the spectra pixel uniform
        print('error spectrum, drop')
        return 0
    one_gal['flux'] = np.array(flux[low_index:up_index])
    one_gal['invar'] = np.array(invar[low_index:up_index])
    one_gal['loglam0'] = loglam_min
    one_gal['pixel'] = input_shape
    return one_gal
    
#====Divide data into 4 pieces, align the data, with the train-validation-test-prediction sample
def Separate_data(fname):
    info = np.load(fname,allow_pickle=True,fix_imports=True)
    z_bin = [info[one_arr] for one_arr in info.files[ : ]]
    data_train,data_val,data_test,data_topre = [],[],[],[]
    
    for onebin in z_bin:   
        data_val = np.concatenate((data_val,onebin[0:val_num]))#===0 to val_num samples as validation
        data_test = np.concatenate((data_test,onebin[val_num:(test_num+val_num)]))#===val_num to test_num+val_num as test
        data_train = np.concatenate((data_train,onebin[(test_num+val_num):(test_num+val_num+tra_num)]))#===test_num+val_num to test_num+val_num+tra_num as train
        data_topre = np.concatenate((data_topre,onebin[0:-1]))#==all as to be predict

    fname = os.path.join(PATH,'tra_data')
    np.save(fname,data_train,allow_pickle=True,fix_imports=True)
    SaveResultcsv(data_train,fname)#===print the value to csv file
             
    fname = os.path.join(PATH,'val_data')
    np.save(fname,data_val,allow_pickle=True,fix_imports=True)
    SaveResultcsv(data_val,fname)#===print the value to csv file
    
    fname = os.path.join(PATH,'test_data')
    np.save(fname,data_test,allow_pickle=True,fix_imports=True)
    SaveResultcsv(data_test,fname)#===print the value to csv file
         
    fname = os.path.join(PATH,'topre_data')
    np.save(fname,data_topre,allow_pickle=True,fix_imports=True)
    SaveResultcsv(data_topre,fname)#===print the value to csv file

#====print the value to csv file
def SaveResultcsv(info,spec_file):
    fname = spec_file+'-num='+str(len(info))#===file name
    saveinfo,temp = [],{}
    for oneinfo in info:
        for indx in oneinfo.keys():
            #if indx != 'flux' or indx != 'invar':#===No save the flux and invar
            temp[indx] = oneinfo[indx] 
        saveinfo.append(temp.copy())#===saving  
    df = pd.DataFrame(saveinfo)
    fname = fname+".csv"
    df.to_csv(fname, mode="w", header=True, index=True)#==also saving the index
    
if __name__=="__main__": 
    #====deviding the selected out sample into 4 piece
    #fname = './Raw_SDSS_Spectra/Galaxy_Spectra_z=0to1_snr=2_num=3.npy'
    fname = './Raw_SDSS_Spectra/Galaxy_Spectra_z=0to1_snr=2_num=1339895.npy'
    fname = Redshift_Bin(fname)#===binnig sample
    Separate_data(fname) #===Divide data into 4 pieces
    
    #=== plot the first 10 picture
    galaxy_spetra_file = os.path.join(PATH,'val_data.npy')
    galaxy_spetra_info = np.load(galaxy_spetra_file,allow_pickle=True,fix_imports=True)
    i = 0
    for one in galaxy_spetra_info:
        Plot_Spectra.Plot_Raw_Spectra(one)
        i = i + 1
        if i > 10:
            break
        