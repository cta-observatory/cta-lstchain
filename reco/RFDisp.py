#!/usr/bin/env python3
'''
Script for DISP reconstruction using Random Forest regressor
'''

from astropy.io import fits 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import math as m
from ctapipe.instrument import OpticsDescription
import astropy.units as u
import ctapipe.coordinates as c
import matplotlib as mpl
import scipy
import disp
import sys
import pandas as pd
import h5py
import h5py.highlevel
from astropy.table import Table


#Read data into pandas DataFrame
filetype = 'hdf5'
filename = "/scratch/bernardos/LST1/Events/gamma_events.hdf5" #File with events
dat = Table.read(filename,format=filetype,path="gamma")
df = dat.to_pandas()

#Get some telescope parameters
tel = OpticsDescription.from_name('LST') #Telescope description
focal_length = tel.equivalent_focal_length.value #Telescope focal length

#Calculate source position and disp distance:
sourcepos = disp.cal_cam_source_pos(df['mc_alt'].get_values(),
                                         df['mc_az'].get_values(),
                                         df['mc_alt_tel'].get_values(),
                                         df['mc_az_tel'].get_values(),
                                         focal_length)
disp = disp.calc_disp(sourcepos[0],
                            sourcepos[1],
                            df['x'].get_values(),
                            df['y'].get_values())
#Add dist and Src position to the DataFrame
df['disp'] = disp
df['src_x'] = sourcepos[0]
df['src_y'] = sourcepos[1]
df = df[abs(df['src_x'])<1e-8]
#df = df[df['intensity'] > 200]
#df = df[abs(df['r']) < 0.9]
# Set Training and Test sets
df['is_train'] = np.random.uniform(0,1,len(df))<= 0.75


#Add some features required for training to the DataFrame
df['w/l'] = df['width']/df['length'] #Width over length
df['mc_energy'] = np.log10(df['mc_energy']*1e3) #Log10(Energy) in GeV
df['intensity'] = np.log10(df['intensity']) #Size in the form log10(size)

train, test = df[df['is_train']==True],df[df['is_train']==False]

#List of features for training
features = ['intensity','time_gradient','width','length','w/l','phi','psi']

#Reconstruct DISP
max_depth = 50
regr_rf = RandomForestRegressor(max_depth=max_depth, random_state=2,n_estimators=100)                                                           
regr_rf.fit(train[features], train['disp'])
disprec = regr_rf.predict(test[features])

#Reconstruct the position.

posdisp = disp.disp_to_Pos(test['disp'],test['x'],test['y'],test['psi'])
theta2_true = (test['src_x']-posdisp[0])**2+(test['src_y']-posdisp[1])**2
posrec = disp.disp_to_Pos(disprec,test['x'],test['y'],test['psi'])
theta2_reco = (test['src_x']-posrec[0])**2+(test['src_y']-posrec[1])**2
plt.hist(theta2_true,bins=50)
plt.yscale('log')
plt.xlabel(r'$\theta^{2}$',fontsize=24)
plt.hist(theta2_reco,bins=50)
plt.show()


difD = ((test['disp']-disprec)/test['disp'])
print(difD.mean(),difD.std())
plt.hist(difD,bins=100,range=[-10,5])
plt.xlabel('$\\frac{disp_{test}-disp_{rec}}{disp_{test}}$',fontsize=30)
plt.figtext(0.6,0.7,'Mean: '+str(round(scipy.stats.describe(difD).mean,6)),fontsize=15)
plt.figtext(0.6,0.65,'Variance: '+str(round(scipy.stats.describe(difD).variance,6)),fontsize=15)
plt.show()

hD = plt.hist2d(test['disp'],disprec,bins=100,range=([0,2],[0,2]))
plt.colorbar(hD[3])
plt.xlabel('$disp_{test}$',fontsize=24)
plt.ylabel('$disp_{rec}$',fontsize=24)
plt.plot(test['disp'],test['disp'],"-",color='red')
plt.show()
