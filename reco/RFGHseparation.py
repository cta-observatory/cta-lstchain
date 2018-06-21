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
intensity = np.append(data_gamma.field('intensity'),
                      data_proton.field('intensity'))
phi = np.append(data_gamma.field('phi'),
                data_proton.field('phi'))
energy = np.append(np.log10(data_gamma.field('mcEnergy')*1e3),
                   np.log10(data_proton.field('mcEnergy')*1e3))
#Log of energy in GeV
x = np.append(data_gamma.field('x'),
              data_proton.field('x'))
y = np.append(data_gamma.field('y'),
              data_proton.field('y'))
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
disp = Disp.calc_DISP(sourcepos[0], sourcepos[1], x, y)

hadroness = np.append(np.zeros(data_gamma.size), np.ones(data_proton.size))

nevents = hadroness.size

X = np.array([intensity, r, width, length, width / length, psi, phi]).T
X_train, X_test, E_train, E_test, D_train, D_test, H_train, H_test = train_test_split(X, energy, disp, hadroness, train_size=int(nevents/2),random_state=4)

#First reconstruct energy

max_depth = 50
regr_rf_e = RandomForestRegressor(max_depth=max_depth, random_state=2)                                                           
regr_rf_e.fit(X_train, E_train)
erec = regr_rf_e.predict(X_test)

#Second, reconstruct DISP:

max_depth = 50
regr_rf_d = RandomForestRegressor(max_depth=max_depth, random_state=2)                                                           
regr_rf_d.fit(X_train, D_train)
disprec = regr_rf_d.predict(X_test)

#Build the new set of training data:
nevents = X_test.shape[0]
nfeatures = X_test.shape[1]

newX = np.zeros((nevents,nfeatures+2))
newX[:,:-2] = X_test

newX = newX.T
newX[nfeatures] = erec
newX[nfeatures+1] = disprec

newX = newX.T

#newX = np.array([intensity,r,width,length,width/length,psi,phi,energy,disp]).T
#newX_train, newX_test, newH_train, newH_test = train_test_split(newX,hadroness,train_size=int(2*nevents/3),random_state=4)

newX_train, newX_test, newH_train, newH_test = train_test_split(newX, H_test,train_size=int(2*nevents/3),random_state=4)

clf = RandomForestClassifier(n_jobs=2,random_state=0)

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

