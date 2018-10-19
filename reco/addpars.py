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
filetype = 'hdf5'
gammafile = "/scratch/bernardos/LST1/Events/gamma_events_point.hdf5" #File with events
protonfile = "/scratch/bernardos/LST1/Events/proton_events_save.hdf5" #File with events

dat_gamma = Table.read(gammafile,format=filetype,path='gamma')
dat_proton = Table.read(protonfile,format=filetype,path='proton')

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

df_gamma['w/l'] = df_gamma['width']/df_gamma['length']
df_proton['w/l'] = df_proton['width']/df_proton['length']

df_gamma.to_hdf("outgammas.hdf5",key="gamma_events")
df_proton.to_hdf("outprotons.hdf5",key="proton_events")
