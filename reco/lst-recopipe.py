import parameters
import reconstruction
from sklearn.externals import joblib

#Simtelarray file with data

datafile = '/scratch/bernardos/LST1/Gamma/Point_Prod-3_LaPalma_flashcam-prod3j/gamma_20deg_0deg_run11716___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel.gz'

#Get out the data from the Simtelarray file:

data = parameters.get_events(datafile,False)

#Load the trained RF for reconstruction:
path_models = "/scratch/bernardos/LST1/Models/"
fileE = path_models+"RFreg_Energy.sav"
fileD = path_models+"RFreg_Disp.sav"
fileH = path_models+"RFcls_GH.sav"

RFreg_Energy = joblib.load(fileE)
RFreg_Disp = joblib.load(fileD)
RFcls_GH = joblib.load(fileH)

#Apply the models to the data
features = ['intensity','time_gradient','width','length','w/l','phi','psi']
reconstruction.ApplyModels(data,features,RFcls_GH,RFreg_Energy,RFreg_Disp)

