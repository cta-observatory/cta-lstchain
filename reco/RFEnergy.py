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
filename = "/home/queenmab/DATA/LST1/Events/gamma_events.hdf5" #File with events
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
features = ['intensity','r','width','length','w/l','phi','psi','impact','mcXmax','mcHfirst']
#features = ['intensity','r','width','length','w/l','phi','psi']
#features = ['intensity','r','length','width']
#Reconstruct Energy
max_depth = 50
regr_rf = RandomForestRegressor(max_depth=max_depth, random_state=2,n_estimators=100)
regr_rf.fit(train[features], train['mcEnergy'])
erec = regr_rf.predict(test[features])

importances = regr_rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in regr_rf.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

print("Feature importances (gini index)")
for f in range(len(features)):
    print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

ordered_features=[]
for index in indices:
    ordered_features=ordered_features+[features[index]]

plt.subplot(221)
plt.title("Feature importances for Energy Reconstruction")
plt.bar(range(len(features)), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(len(features)), ordered_features)
plt.xlim([-1, len(features)])


#Plot Energy and Disp reconstructions
plt.subplot(222)
difE = (((test['mcEnergy']-erec)/test['mcEnergy'])*np.log10(10))
print(difE.mean(),difE.std())
plt.hist(difE,bins=100)
plt.xlabel('$\\frac{E_{test}-E_{rec}}{E_{test}}$')
plt.figtext(0.6,0.7,'Mean: '+str(round(scipy.stats.describe(difE).mean,6)))
plt.figtext(0.6,0.65,'Variance: '+str(round(scipy.stats.describe(difE).variance,6)))

plt.subplot(223)
hE = plt.hist2d(test['mcEnergy'],erec,bins=100)
plt.colorbar(hE[3])
plt.xlabel('$log_{10}E_{test}$')
plt.ylabel('$log_{10}E_{rec}$')
plt.plot(test['mcEnergy'],test['mcEnergy'],"-",color='red')
plt.show()
