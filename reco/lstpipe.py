import numpy as np
import pandas as pd
import transformations
import parameters
import reconstruction

#Simtelarray file with data

datafile = '/scratch/bernardos/LST1/Gamma/Point_Prod-3_LaPalma_flashcam-prod3j/gamma_20deg_0deg_run11716___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel.gz'

#Files with MC events for training the RF

gammafile = "/scratch/bernardos/LST1/Events/gamma_events_point.hdf5" #File with events
protonfile = "/scratch/bernardos/LST1/Events/proton_events.hdf5" #File with events

#Get out the data from the Simtelarray file:

data = parameters.get_events(datafile,False)

#Train the models

features = ['intensity','time_gradient','width','length','w/l','phi','psi']

RFreg_Energy,RFreg_Disp,RFcls_GH = reconstruction.buildModels(gammafile,protonfile,features,True)

#Apply the models to the data

reconstruction.ApplyModels(data,features,RFcls_GH,RFreg_Energy,RFreg_Disp)


