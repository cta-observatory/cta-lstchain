#!/usr/bin/env python3
'''
Script for Energy reconstruction using Random Forest regressor
'''

from astropy.io import fits
import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
import math as m
from ctapipe.instrument import OpticsDescription
import astropy.units as u
import ctapipe.coordinates as c
import matplotlib as mpl
import Disp
import sys

hdu_gamma = fits.open("/home/queenmab/DATA/LST1/Events/Gamma_events.fits") #File with events
hdu_proton = fits.open("/home/queenmab/DATA/LST1/Events/Proton_events.fits") #File with events

data_gamma = hdu_gamma[1].data
data_proton = hdu_proton[1].data

tel = OpticsDescription.from_name('LST') #Telescope description
focal_length = tel.equivalent_focal_length.value #Telescope focal length

data_gamma = hdu_gamma[1].data
data_proton = hdu_proton[1].data
disp = np.array([]) #Disp quantity

width = np.append(data_gamma.field('width'),
                  data_proton.field('width'))
length = np.append(data_gamma.field('length'),
                   data_proton.field('length'))
size = np.append(data_gamma.field('size'),
                 data_proton.field('size'))
phi = np.append(data_gamma.field('phi'),
                data_proton.field('phi'))
energy = np.append(np.log10(data_gamma.field('mcEnergy')*1e3),
                   np.log10(data_proton.field('mcEnergy')*1e3))
#Log of energy in GeV
cen_x = np.append(data_gamma.field('cen_x'),
                  data_proton.field('cen_x'))
cen_y = np.append(data_gamma.field('cen_y'),
                  data_proton.field('cen_y'))
psi = np.append(data_gamma.field('psi'),
                data_proton.field('psi'))
r = np.append(data_gamma.field('r'),
              data_proton.field('r'))

mcAlt = np.append(data_gamma.field('mcAlt'),
                  data_proton.field('mcAlt'))
mcAz = np.append(data_gamma.field('mcAz'),
                 data_proton.field('mcAz'))
mcAlttel = np.append(data_gamma.field('mcAlttel'),
                     data_proton.field('mcAlttel'))
mcAztel = np.append(data_gamma.field('mcAztel'),
                    data_proton.field('mcAztel'))

sourcepos = Disp.calc_CamSourcePos(mcAlt,mcAz,mcAlttel,mcAztel,focal_length)
disp = Disp.calc_DISP(sourcepos[0],sourcepos[1],cen_x,cen_y)

hadroness = np.append(np.zeros(data_gamma.size),np.ones(data_proton.size))

nevents = hadroness.size

X = np.array([size,r,width,length,width/length,psi,phi]).T
X_train, X_test, E_train, E_test, D_train, D_test, H_train, H_test = train_test_split(X, energy, disp, hadroness, train_size=int(nevents/2),random_state=4)

#First reconstruct energy

max_depth = 50
regr_rf_e = RandomForestRegressor(max_depth=max_depth, random_state=2,n_estimators=100)                                                           
regr_rf_e.fit(X_train, E_train)
erec = regr_rf_e.predict(X_test)

difE = ((E_test-erec)/E_test*np.log10(10))
print(difE.mean(),difE.std())
plt.hist(difE,bins=100)
plt.xlabel('$\\frac{E_{test}-E_{rec}}{E_{test}}$',fontsize=30)
plt.figtext(0.6,0.7,'Mean: '+str(round(scipy.stats.describe(difE).mean,6)),fontsize=15)
plt.figtext(0.6,0.65,'Variance: '+str(round(scipy.stats.describe(difE).variance,6)),fontsize=15)
plt.show()

#figE, ax = plt.subplots()
hE = plt.hist2d(E_test,erec,bins=50)
plt.colorbar(hE[3])
plt.xlabel('$log_{10}E_{test}$',fontsize=24)
plt.ylabel('$log_{10}E_{rec}$',fontsize=24)
plt.show()


#Second, reconstruct DISP:

max_depth = 50
regr_rf_d = RandomForestRegressor(max_depth=max_depth, random_state=2,n_estimators=100)                                                           
regr_rf_d.fit(X_train, D_train)
disprec = regr_rf_d.predict(X_test)

difD = ((D_test-disprec))
print(difD.mean(),difD.std())
plt.hist(difD,bins=100)
plt.xlabel('$Disp_{test} - Disp_{rec}$',fontsize=24)
plt.figtext(0.6,0.7,'Mean: '+str(round(scipy.stats.describe(difD).mean,6)),fontsize=15)
plt.figtext(0.6,0.65,'Variance: '+str(round(scipy.stats.describe(difD).variance,6)),fontsize=15)
plt.show()

hD = plt.hist2d(D_test,disprec,bins=50)
plt.colorbar(hD[3])
plt.xlabel('$Disp_{test}$',fontsize=24)
plt.ylabel('$Disp_{rec}$',fontsize=24)
plt.show()

#Build the new set of training data:

Energy_cut = 2.699

nevents = X_test.shape[0]
nfeatures = X_test.shape[1]

newX = np.zeros((nevents,nfeatures+2))
newX[:,:-2] = X_test

newX = newX.T
newX[nfeatures] = erec
newX[nfeatures+1] = disprec

newX = newX.T

#newX = np.array([size,r,width,length,width/length,psi,phi,energy,disp]).T
#newX_train, newX_test, newH_train, newH_test = train_test_split(newX,hadroness,train_size=int(2*nevents/3),random_state=4)

nevents = X_test[erec > Energy_cut].shape[0]

newX_train, newX_test, newH_train, newH_test = train_test_split(newX[erec > Energy_cut], H_test[erec > Energy_cut],train_size=int(2*nevents/3),random_state=4)


clf = RandomForestClassifier(max_depth = 50,n_jobs=50,random_state=4, n_estimators=1000)

clf.fit(newX_train,newH_train)

result = clf.predict(newX_test)

check = clf.predict_proba(newX_test)[0:,1]

accuracy = accuracy_score(newH_test, result)
print(accuracy)

fpr_rf, tpr_rf, _ = roc_curve(newH_test, check)

plt.plot(fpr_rf, tpr_rf, label='Reco Energy and Disp')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc='best')
plt.show()

