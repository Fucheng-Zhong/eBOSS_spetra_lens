# eBOSS_spetra_lens
# The source code of finding strong galaxy-galaxy lens from SDSS spectra (arxiv: 2202.08439)

# Total five-step of this project.

## 1. You need to run "Serching_Galaxy.py" to select out all possible useable galaxies spectra. Notice, you need to download the SDSS spectra "spPlate.fits" and "spZbest.fits" files to the "PATH" folder.

## 2. Run "SDSS_Spetra_Produce.py" to Bin the redshift for spectra generated in step.1, and cut the spectra in a uniform format, which same as the shape of CNN models. It will produce 4 pieces of data, corresponding train, validation, test, and prediction samples.

## 3. According to the output data of step.3, run the "Build_CNN_Data.py" to generate CNN's labeled positive and negative samples. The positive samples will be adding artificial emission lines.

## 4. Run the "model.py" to train the CNNs.

## 5. After training CNNs' modes, run "pred.py" to predict the value of PL, ZE, and ZG. Predict results will be saved in csv files.
