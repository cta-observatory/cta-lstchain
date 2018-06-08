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

X_e = np.array([size,r,width,length,width/length,psi,phi]).T
X_etrain, X_etest, E_train, E_test = train_test_split(X_e, energy,
                                                    train_size=int(2*nevents/3),
                                                      random_state=4)

max_depth = 50
regr_rf = RandomForestRegressor(max_depth=max_depth, random_state=2)                                                           
regr_rf.fit(X_etrain, E_train)
erec = regr_rf.predict(X_etest)

difE = ((E_test-erec)*np.log10(10))
print(difE.mean(),difE.std())
plt.hist(difE,bins=100)
plt.xlabel('$\\frac{E_{test}-E_{rec}}{E_{test}}$',fontsize=30)
plt.figtext(0.6,0.7,'Mean: '+str(round(scipy.stats.describe(difE).mean,6)),fontsize=15)
plt.figtext(0.6,0.65,'Variance: '+str(round(scipy.stats.describe(difE).variance,6)),fontsize=15)
plt.show()

figE, ax = plt.subplots()
hE = ax.hist2d(E_test,erec,bins=50)
plt.colorbar(hE[3],ax=ax)
plt.xlabel('$log_{10}E_{test}$',fontsize=24)
plt.ylabel('$log_{10}E_{rec}$',fontsize=24)
figE.show()
