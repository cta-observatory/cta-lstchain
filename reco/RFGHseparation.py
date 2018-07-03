#!/usr/bin/env python3
'''
Script for G/H separation using Ranfom Forest Classifier
Energy and Disp are reconstructed usign Random Forest Regressor
'''

from astropy.io import fits
import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
import math as m
from ctapipe.instrument import OpticsDescription
import astropy.units as u
import ctapipe.coordinates as c
import matplotlib as mpl
import Disp
import sys
import pandas as pd
from astropy.table import Table
import h5py
import h5py.highlevel

#Read data into pandas DataFrame
filetype = 'hdf5'
gammafile = "/scratch/bernardos/LST1/Events/gamma_events.hdf5" #File with events
protonfile = "/scratch/bernardos/LST1/Events/proton_events.hdf5" #File with events
dat_gamma = Table.read(gammafile,format=filetype)
dat_proton = Table.read(protonfile,format=filetype)

df_gamma = dat_gamma.to_pandas()
df_proton = dat_proton.to_pandas()

#Get some telescope parameters
tel = OpticsDescription.from_name('LST') #Telescope description
focal_length = tel.equivalent_focal_length.value #Telescope focal length

#Calculate source position and Disp distance:
sourcepos_gamma = Disp.calc_CamSourcePos(df_gamma['mcAlt'].get_values(),
                                         df_gamma['mcAz'].get_values(),
                                         df_gamma['mcAlttel'].get_values(),
                                         df_gamma['mcAztel'].get_values(),
                                         focal_length)
disp_gamma = Disp.calc_DISP(sourcepos_gamma[0],
                            sourcepos_gamma[1],
                            df_gamma['x'].get_values(),
                            df_gamma['y'].get_values())

sourcepos_proton = Disp.calc_CamSourcePos(df_proton['mcAlt'].get_values(),
                                         df_proton['mcAz'].get_values(),
                                         df_proton['mcAlttel'].get_values(),
                                         df_proton['mcAztel'].get_values(),
                                         focal_length)
disp_proton = Disp.calc_DISP(sourcepos_proton[0],
                            sourcepos_proton[1],
                            df_proton['x'].get_values(),
                            df_proton['y'].get_values())

#Add dist to the DataFrame
df_gamma['disp'] = disp_gamma
df_proton['disp'] = disp_proton
df_gamma['hadroness'] = np.zeros(df_gamma.shape[0])
df_proton['hadroness'] = np.ones(df_proton.shape[0])

# Set Training and Test sets
df_gamma['is_train'] = np.random.uniform(0,1,len(df_gamma))<= 0.5
df_proton['is_train'] = np.random.uniform(0,1,len(df_proton))<= -1.
df = df_gamma.append(df_proton,ignore_index=True)

#Add some features required for training to the DataFrame
df['w/l'] = df['width']/df['length'] #Width over length
df['mcEnergy'] = np.log10(df['mcEnergy']*1e3) #Log10(Energy) in GeV
df['intensity'] = np.log10(df['intensity']) #Size in the form log10(size)

#Create a training set only with gammas and a test set with gammas and protons
train_gammas, test = df[(df['is_train']==True) & (df['hadroness']==0)],df[df['is_train']==False]

#List of features for training
features = ['intensity','r','width','length','w/l','phi','psi','impact']
#features = ['size','r','mcHfirst','width','length','w/l','phi','psi'] #Here we include true mcHfirst for testing

max_depth = 50
regr_rf_e = RandomForestRegressor(max_depth=max_depth, random_state=2,n_estimators=100)                                                           
regr_rf_e.fit(train_gammas[features], train_gammas['mcEnergy'])
erec = regr_rf_e.predict(test[features])

#Reconstruct Disp
regr_rf_disp = RandomForestRegressor(max_depth=max_depth, random_state=2,n_estimators=100)                                                           
regr_rf_disp.fit(train_gammas[features], train_gammas['disp'])
disprec = regr_rf_disp.predict(test[features])

#Plot Energy and Disp reconstructions
difE = (((test['mcEnergy']-erec)/test['mcEnergy'])*np.log10(10))
print(difE.mean(),difE.std())
plt.hist(difE,bins=100)
plt.xlabel('$\\frac{E_{test}-E_{rec}}{E_{test}}$',fontsize=30)
plt.figtext(0.6,0.7,'Mean: '+str(round(scipy.stats.describe(difE).mean,6)),fontsize=15)
plt.figtext(0.6,0.65,'Variance: '+str(round(scipy.stats.describe(difE).variance,6)),fontsize=15)
plt.show()

hE = plt.hist2d(test['mcEnergy'],erec,bins=100)
plt.colorbar(hE[3])
plt.xlabel('$log_{10}E_{test}$',fontsize=24)
plt.ylabel('$log_{10}E_{rec}$',fontsize=24)
plt.plot(test['mcEnergy'],test['mcEnergy'],"-",color='red')
plt.show()

difD = ((test['disp']-disprec)/test['disp'])
print(difD.mean(),difD.std())
plt.hist(difD,bins=100,range=[-10,5])
plt.xlabel('$\\frac{Disp_{test}-Disp_{rec}}{Disp_{test}}$',fontsize=30)
plt.figtext(0.6,0.7,'Mean: '+str(round(scipy.stats.describe(difD).mean,6)),fontsize=15)
plt.figtext(0.6,0.65,'Variance: '+str(round(scipy.stats.describe(difD).variance,6)),fontsize=15)
plt.show()

hD = plt.hist2d(test['disp'],disprec,bins=100,range=([0,2],[0,2]))
plt.colorbar(hD[3])
plt.xlabel('$Disp_{test}$',fontsize=24)
plt.ylabel('$Disp_{rec}$',fontsize=24)
plt.plot(test['disp'],test['disp'],"-",color='red')
plt.show()

#Perform Gamma/Hadron separation:

#Now we build a new training set with reconstructed Energy and Disp:

#Set a cut in energy, 0 for no cuts.
Energy_cut = 3.

test['Erec'] = erec
test['Disprec'] = disprec

test = test[test['mcEnergy']>Energy_cut]

#Select the training/test events
test['is_train'] = np.random.uniform(0,1,len(test))<= 0.75

#Build new train/test sets:
train,test = test[test['is_train']==True],test[test['is_train']==False]

features = ['Erec','Disprec','intensity','r','width','length','w/l','phi','psi','impact']


#Classify Gamma/Hadron
clf = RandomForestClassifier(max_depth = 50,
                             n_jobs=10,
                             random_state=4,
                             n_estimators=500)

clf.fit(train[features],train['hadroness'])
result = clf.predict(test[features])

# Plot ROC curve:
check = clf.predict_proba(test[features])[0:,1]
accuracy = accuracy_score(test['hadroness'], result)
print(accuracy)

fpr_rf, tpr_rf, _ = roc_curve(test['hadroness'], check)

plt.plot(fpr_rf, tpr_rf, label='Energy Cut: '+'%.3f'%(pow(10,Energy_cut)/1000)+' TeV')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc='best')
plt.show()


