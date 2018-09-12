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
gammafile = "/home/queenmab/DATA/LST1/Events/gamma_events_point.hdf5" #File with events
protonfile = "/home/queenmab/DATA/LST1/Events/proton_events_diff.hdf5" #File with events
dat_gamma = Table.read(gammafile,format=filetype,path="gamma")
dat_proton = Table.read(protonfile,format=filetype,path="proton")

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
df_gamma['SrcX'] = sourcepos_gamma[0]
df_gamma['SrcY'] = sourcepos_gamma[1]
df_proton['SrcX'] = sourcepos_proton[0]
df_proton['SrcY'] = sourcepos_proton[1] 
df_gamma['disp'] = disp_gamma
df_proton['disp'] = disp_proton
df_gamma['hadroness'] = np.zeros(df_gamma.shape[0])
df_proton['hadroness'] = np.ones(df_proton.shape[0])

#Cut events in the border:
df_gamma = df_gamma[abs(df_gamma['r'])<0.94]
df_proton = df_proton[abs(df_proton['r'])<0.94]

#Cut showers with low intensity
df_gamma = df_gamma[abs(df_gamma['intensity'])>60]
df_proton = df_proton[abs(df_proton['intensity'])>60]

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
features = ['intensity','r','width','length','w/l','phi','psi']
#features = ['intensity','r','width','length']


max_depth = 50
regr_rf_e = RandomForestRegressor(max_depth=max_depth, random_state=2,n_estimators=100)                                                           
regr_rf_e.fit(train_gammas[features], train_gammas['mcEnergy'])
erec = regr_rf_e.predict(test[features])

#Reconstruct Disp
regr_rf_disp = RandomForestRegressor(max_depth=max_depth, random_state=2,n_estimators=100)                                                           
regr_rf_disp.fit(train_gammas[features], train_gammas['disp'])
disprec = regr_rf_disp.predict(test[features])

#Reconstruct position of the Source from Disp

posdisp = Disp.Disp_to_Pos(test['disp'],test['x'],test['y'],test['psi'])
theta2_true = (test['SrcX']-posdisp[0])**2+(test['SrcY']-posdisp[1])**2
posrec = Disp.Disp_to_Pos(disprec,test['x'],test['y'],test['psi'])
theta2_reco = (test['SrcX']-posrec[0])**2+(test['SrcY']-posrec[1])**2

test['theta2_true'] = theta2_true
test['theta2_reco'] = theta2_reco

#Plot Energy and Disp reconstructions
plt.subplot(121)
difE = (((test['mcEnergy']-erec)/test['mcEnergy'])*np.log10(10))
print(difE.mean(),difE.std())
plt.hist(difE,bins=100)
plt.xlabel('$\\frac{E_{test}-E_{rec}}{E_{test}}$',fontsize=15)
plt.figtext(0.6,0.7,'Mean: '+str(round(scipy.stats.describe(difE).mean,6)),fontsize=15)
plt.figtext(0.6,0.65,'Variance: '+str(round(scipy.stats.describe(difE).variance,6)),fontsize=15)

plt.subplot(122)
hE = plt.hist2d(test['mcEnergy'],erec,bins=100)
plt.colorbar(hE[3])
plt.xlabel('$log_{10}E_{test}$',fontsize=15)
plt.ylabel('$log_{10}E_{rec}$',fontsize=15)
plt.plot(test['mcEnergy'],test['mcEnergy'],"-",color='red')
plt.show()

plt.subplot(221)
difD = ((test['disp']-disprec)/test['disp'])
print(difD.mean(),difD.std())
plt.hist(difD,bins=100,range=[-2,1.5])
plt.xlabel('$\\frac{Disp_{test}-Disp_{rec}}{Disp_{test}}$',fontsize=15)
plt.figtext(0.6,0.7,'Mean: '+str(round(scipy.stats.describe(difD).mean,6)),fontsize=15)
plt.figtext(0.6,0.65,'Variance: '+str(round(scipy.stats.describe(difD).variance,6)),fontsize=15)

plt.subplot(222)
hD = plt.hist2d(test['disp'],disprec,bins=100,range=([0,2],[0,2]))
plt.colorbar(hD[3])
plt.xlabel('$Disp_{test}$',fontsize=15)
plt.ylabel('$Disp_{rec}$',fontsize=15)
plt.plot(test['disp'],test['disp'],"-",color='red')

plt.subplot(223)
plt.hist(test['theta2_true'],bins=50,range=[0,60],histtype=u'step',label=r'With Hillas Disp')
plt.hist(test['theta2_reco'],bins=50,range=[0,60],histtype=u'step',label=r'With Reconstructed Disp')
plt.yscale('log')
plt.legend()
plt.xlabel(r'$\theta^{2}$',fontsize=15)
plt.ylabel(r'# of Gamma and Proton events',fontsize=15)

plt.subplot(224)
plt.hist(test[test['hadroness']<1]['theta2_true'],bins=50,range=[0,20],histtype=u'step',label=r'With Hillas Disp')
plt.hist(test[test['hadroness']<1]['theta2_reco'],bins=50,range=[0,20],histtype=u'step',label=r'With Reconstructed Disp')
plt.yscale('log')
plt.legend()
plt.xlabel(r'$\theta^{2}$',fontsize=15)
plt.ylabel(r'# of Only Gamma events',fontsize=15)

plt.show()

#Perform Gamma/Hadron separation:

#Now we build a new training set with reconstructed Energy and Disp:

#Set a cut in energy, 0 for no cuts.
Energy_cut = -1.

test['Erec'] = erec
test['Disprec'] = disprec

test = test[test['mcEnergy']>Energy_cut]

#Select the training/test events
test['is_train'] = np.random.uniform(0,1,len(test))<= 0.75

#Build new train/test sets:
train,test = test[test['is_train']==True],test[test['is_train']==False]

#features = ['intensity','r','width','length','w/l','phi','psi','impact','mcXmax','mcHfirst']
features = ['Erec','intensity','width','length','w/l','phi','psi']

#Classify Gamma/Hadron
clf = RandomForestClassifier(max_depth = 50,
                             n_jobs=10,
                             random_state=4,
                             n_estimators=500)

clf.fit(train[features],train['hadroness'])
result = clf.predict(test[features])

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

print("Feature importances (gini index)")
for f in range(len(features)):
    print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

ordered_features=[]
for index in indices:
    ordered_features=ordered_features+[features[index]]

plt.subplot(121)
plt.title("Feature importances for G/H separation",fontsize=15)
plt.bar(range(len(features)), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(len(features)), ordered_features)
plt.xlim([-1, len(features)])

# Plot ROC curve:
check = clf.predict_proba(test[features])[0:,1]
accuracy = accuracy_score(test['hadroness'], result)
print(accuracy)

fpr_rf, tpr_rf, _ = roc_curve(test['hadroness'], check)

plt.subplot(122)
plt.plot(fpr_rf, tpr_rf, label='Energy Cut: '+'%.3f'%(pow(10,Energy_cut)/1000)+' TeV')
plt.xlabel('False positive rate',fontsize=15)
plt.ylabel('True positive rate',fontsize=15)
plt.legend(loc='best')
plt.show()

plt.hist(test[test['hadroness']<1]['theta2_true'],bins=50,range=[0,2],histtype=u'step',label =r'With Hillas Disp')
plt.hist(test[test['hadroness']<1]['theta2_reco'],bins=50,range=[0,2],histtype=u'step',label =r'With Reconstructed Disp')
plt.yscale('log')
plt.xlabel(r'$\theta^{2}$',fontsize=15)
plt.ylabel(r'# Gamma events (after g/h separation)',fontsize=15)
plt.legend()
plt.show()
