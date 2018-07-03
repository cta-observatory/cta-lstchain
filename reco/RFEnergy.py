#!/usr/bin/env python3
'''
Script for Energy reconstruction using Random Forest regressor
'''

from astropy.io import fits
import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import math as m
from ctapipe.instrument import OpticsDescription
import astropy.units as u
import ctapipe.coordinates as c
import matplotlib as mpl
import Disp
import sys
import h5py
import h5py.highlevel
import pandas as pd
from astropy.table import Table

#Read data into pandas DataFrame

filetype = 'hdf5'
filename = "/scratch/bernardos/LST1/Events/gamma_events.hdf5" #File with events
dat = Table.read(filename,format=filetype)


df = dat.to_pandas()

#Get some telescope parameters
tel = OpticsDescription.from_name('LST') #Telescope description
focal_length = tel.equivalent_focal_length.value #Telescope focal length

#Calculate source position and Disp distance:
sourcepos = Disp.calc_CamSourcePos(df['mcAlt'].get_values(),
                                         df['mcAz'].get_values(),
                                         df['mcAlttel'].get_values(),
                                         df['mcAztel'].get_values(),
                                         focal_length)
disp = Disp.calc_DISP(sourcepos[0],
                            sourcepos[1],
                            df['x'].get_values(),
                            df['y'].get_values())
#Add dist to the DataFrame
df['disp'] = disp

# Set Training and Test sets
df['is_train'] = np.random.uniform(0,1,len(df))<= 0.75

#Add some features required for training to the DataFrame
df['w/l'] = df['width']/df['length'] #Width over length
df['mcEnergy'] = np.log10(df['mcEnergy']*1e3) #Log10(Energy) in GeV
df['intensity'] = np.log10(df['intensity']) #Size in the form log10(size)

train, test = df[df['is_train']==True],df[df['is_train']==False]

#List of features for training
features = ['intensity','r','width','length','w/l','phi','psi','impact']

#Reconstruct Energy
max_depth = 50
regr_rf = RandomForestRegressor(max_depth=max_depth, random_state=2,n_estimators=100)                                                           
regr_rf.fit(train[features], train['mcEnergy'])
erec = regr_rf.predict(test[features])

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
