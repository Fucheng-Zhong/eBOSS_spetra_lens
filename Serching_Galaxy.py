'''
For serching a set of galaxy spectrum
with some SNR and redshift condiction
'''
from unicodedata import name
from astropy.io import fits as pyfits
import numpy as np
import os,Plot_Spectra

#===slected spectra redshif range and minmal SNR
z_min, z_max, SNR_min = 0, 1, 2

#===the saving location of SDSS specta data
PATH = './Raw_SDSS_Spectra'
if not os.path.exists(PATH):
        os.makedirs(PATH)

#===notice !!!!
#===we have download all DR16 spectra data in server!!!
#===now we just run for loop to pick out the spectra meet the condiction

#=====obtain the all spZbest files' name in dir
def get_file_name(Date_Path):
    spZbest_list = []
    file_name_list = os.listdir(Date_Path)
    for i in range(len(file_name_list)):
        extend_name = file_name_list[i].rsplit('.', 1)[-1]
        if extend_name == 'fits':    #comfirm it's .fits file
            first_file_name = file_name_list[i].split('-', 1)[0]
            if first_file_name == 'spZbest': #spZbest it's spZbest file
                spZbest_list.append(file_name_list[i])
    return spZbest_list

#=====obtain the spPlate_file file's name
def get_plate_mjd(spZbest_name):
    temp = spZbest_name.rsplit('.', 1)[-2]
    plate = temp.split('-', 2)[1]
    mjd = temp.split('-', 2)[2]
    #print('palte:',plate,'mjd:',mjd)
    spPlate_file = 'spPlate-' + plate +'-'+ mjd + '.fits'
    return spPlate_file

#==========read out the spectra from .fits files
def Select_Out_Galaxy(spZbest_file,spPlate_file):
    with pyfits.open(spPlate_file) as readin_spPlate:
        hdr0=readin_spPlate[0].header
        loglam0=hdr0['COEFF0']    #loglambda star point
        loglam_step=hdr0['COEFF1']#loglambda step
        naxis1=hdr0['NAXIS1']    #spectra pixel number
        naxis2=hdr0['NAXIS2']    #spectra number
        flux=readin_spPlate[0].data #flux
        invar=readin_spPlate[1].data #invar
        
        hdr5 = readin_spPlate[5].data
        RA = hdr5.RA
        DEC = hdr5.DEC
        
    with pyfits.open(spZbest_file) as readin_spZbest:
        data=readin_spZbest[1].data
        PLATE=data.PLATE #plate-MJD-fiber
        FIBERID=data.FIBERID 
        MJD=data.MJD 
        Z=data.Z   #best z
        CLASS=data.CLASS  #object class
        OBJTYPE=data.OBJTYPE
        SUBCLASS=data.SUBCLASS
        #RA = data.PLUG_RA
        #DEC = data.PLUG_DEC

    #====begin serching
    galaxy_selected = []
    for i in range(naxis2): 
        tar_class=CLASS[i].decode('utf-8') # in server, you need to decode
        tar_type=OBJTYPE[i].decode('utf-8')
        #tar_type=OBJTYPE[i]
        Z_one=Z[i]
        #=====redshift out of range
        if Z_one<=z_min or Z_one>z_max:
            continue
        #is it galaxy ？ 
        if tar_type.split(' ')[0] == 'GALAXY': 
            #====calcuate spectrum mean SNR
            sigma=np.sqrt(1./np.array(invar[i]+1e-6))
            sn_mean=sum(flux[i]/sigma)/len(flux[i])
            if sn_mean >= SNR_min:
                #====save as dict
                ONE_galaxy = {'RA':RA[i],'DEC':DEC[i],'PLATE':PLATE[i],'MJD':MJD[i],'FIBER':FIBERID[i],'ZG':Z[i],
                              'flux':flux[i].copy(),'invar':invar[i].copy(),'loglam0':loglam0,'loglam_step':loglam_step,
                              'pixel':naxis1,'ZE':0,'sn_mean':sn_mean}
                print('#Find one galaxy#','RA:',ONE_galaxy['RA'],'DEC:',ONE_galaxy['DEC'],'sn_maen:',sn_mean,'ZG:',ONE_galaxy['ZG'])  
                galaxy_selected.append(ONE_galaxy)
    #====none matching
    if len(galaxy_selected) == 0:
        print('No one get!')
        return []
    return galaxy_selected

#===serching the spectra and saving
def Slect_galaxy_all(Date_Path):
    print('staring get file list...')
    spZbest_list = get_file_name(Date_Path) #obtain all spZbest files' name
    count, remain = 0,len(spZbest_list)
    print('Total spZbest file:',remain)
    
    galaxy_INFO = []
    for target in spZbest_list:
        remain = remain - 1
        print(str(Date_Path)+'serching file:' + target,'###galaxy speturm num:',len(galaxy_INFO))
        spZbest_file = target
        spPlate_file = get_plate_mjd(spZbest_file)
        spZbest_file = os.path.join(Date_Path,spZbest_file)
        spPlate_file = os.path.join(Date_Path,spPlate_file)
        #=====exist corresponding spPlate_file?
        if os.path.exists(spPlate_file) == False:
            print('NO such flie:',spPlate_file)
            continue
        #=========select galaxy spectra from this plate & mjd files
        ONE_galaxy = Select_Out_Galaxy(spZbest_file,spPlate_file)
        if ONE_galaxy == []: # none galaxy
            continue
        else:
            for one in ONE_galaxy:
                galaxy_INFO.append(one)
            count = len(galaxy_INFO)
        print('Remain fit file:',remain,'\nTotal galaxies:',count)
    print('End Serching','Total galaxy:',len(galaxy_INFO),)
    #===saving
    if len(galaxy_INFO) > 0:
        head_name = 'Galaxy_Spectra_z='+str(z_min)+'to'+str(z_max)+'_snr='+str(SNR_min)
        file_name = head_name+'_num='+str(count)
        file_name = os.path.join(PATH,file_name)+'.npy'
        #====save as a .npy file
        galaxy_INFO = np.array(galaxy_INFO)
        np.save(file_name, galaxy_INFO,allow_pickle=True,fix_imports=True)
        print('saving:',file_name)
        return file_name

#=== runing code ===#
if __name__ == '__main__':
    #Slect_galaxy_all('../DR6/PATH')
    #Slect_galaxy_all('../DR9/PATH')
    galaxy_spetra_file=Slect_galaxy_all('../DR16/PATH') #your data
    #galaxy_spetra_file = Slect_galaxy_all('./PATH')
    #===loading the picked out spectra
    galaxy_spetra_info = np.load(galaxy_spetra_file,allow_pickle=True,fix_imports=True)
    #===plot 10 spectra...
    i = 0
    for one in galaxy_spetra_info:
        Plot_Spectra.Plot_Raw_Spectra(one)
        i = i + 1
        if i > 10:
            break