import numpy as np
import pandas as pd
import reconstruction

#Files with events for training
gammafile = "/scratch/bernardos/LST1/Events/gamma_events_point.hdf5" #File with events
protonfile = "/scratch/bernardos/LST1/Events/proton_events.hdf5" #File with events

#Train the models
path_models = "/scratch/bernardos/LST1/Models/"
features = ['intensity','time_gradient','width','length','w/l','phi','psi']
RFreg_Energy,RFreg_Disp,RFcls_GH = reconstruction.buildModels(gammafile,protonfile,features,True,path_models)

