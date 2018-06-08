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
import Disp


hdu_list = fits.open("/home/queenmab/DATA/LST1/Events/events.fits") #File with events
hdu_list[1].data

tel = OpticsDescription.from_name('LST') #Telescope description
focal_length = tel.equivalent_focal_length.value #Telescope focal length

nevents = hdu_list[1].data.field(0).size #Total number of events
disp = np.array([]) #Disp quantity

data = hdu_list[1].data

width = data.field('width')
length = data.field('length')
size = data.field('size')
phi = data.field('phi')
energy = np.log10(data.field('mcEnergy')*1e3) #Log of energy in GeV
cen_x = data.field('cen_x')
cen_y = data.field('cen_y')
psi = data.field('psi')
r = data.field('r')

mcAlt = data.field('mcAlt')
mcAz = data.field('mcAz')
mcAlttel = data.field('mcAlttel')
mcAztel = data.field('mcAztel')

sourcepos = Disp.calc_CamSourcePos(mcAlt,mcAz,mcAlttel,mcAztel,focal_length)
disp = Disp.calc_DISP(sourcepos[0],sourcepos[1],cen_x,cen_y)

max_depth = 50
regr_rf = RandomForestRegressor(max_depth=max_depth, random_state=2)

X_disp = np.array([width/length,size,width,length,dist,phi,psi]).T
X_dtrain, X_dtest, Disp_train, Disp_test = train_test_split(X_disp, disp,
                                                    train_size=int(2*nevents/3),
                                                    random_state=4)

regr_rf.fit(X_dtrain, Disp_train)
Disprec = regr_rf.predict(X_dtest)

difD = ((Disp_test-Disprec))
print(difD.mean(),difD.std())
print(Disp_test.mean(),Disp_test.std())
print(Disprec.mean(),Disprec.std())
plt.hist(difD,bins=100)
plt.xlabel('$Disp_{test} - Disp_{rec}$',fontsize=24)
plt.figtext(0.6,0.7,'Mean: '+str(round(scipy.stats.describe(difD).mean,6)),fontsize=15)
plt.figtext(0.6,0.65,'Variance: '+str(round(scipy.stats.describe(difD).variance,6)),fontsize=15)
plt.show()

figD, aax = plt.subplots()
hD = aax.hist2d(Disp_test,Disprec,bins=50)
plt.colorbar(hD[3],ax=aax)
plt.xlabel('$Disp_{test}$',fontsize=24)
plt.ylabel('$Disp_{rec}$',fontsize=24)
figD.show()
