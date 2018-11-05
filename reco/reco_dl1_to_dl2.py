"""Module with functions for Energy and Disp reconstruction and G/H
separation. There are functions for raining random forest and for
applying them to data. The RF can be saved into a file for later use.

Usage:

"import reco_dl1_to_dl2"
"""
import numpy as np
import pandas as pd
import utils
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.externals import joblib

def split_traintest(data,proportion):
    """
    Split a dataset in "train" and "test" sets.

    Parameters:
    -----------
    data: pandas DataFrame

    proportion: float
    Percentage of the total dataset that will be part of the train set.

    Returns:
    --------
    pandas dataFrame: train
    
    pandas dataFrame: test
    
    """
    data['is_train'] = np.random.uniform(0,1,len(data))<= proportion
    train = data[(data['is_train']==True)]
    test = data[(data['is_train']==False)]
    return train,test

def trainRFreco(train,features):
    """
    Trains two Random Forest regressors for Energy and Disp
    reconstruction respectively. Returns the trained RF.

    Parameters:
    -----------
    train: pandas DataFrame
    data set for training the RF

    features: list of strings
    List of features to train the RF
    
    Returns:
    --------
    RandomForestRegressor: regr_rf_e

    RandomForestRegressor: regr_rf_disp
    """

    print("Given features: ",features)
    print("Number of events for training: ",train.shape[0])
    print("Training Random Forest Regressor for Energy Reconstruction...")
    
    
    max_depth = 50
    regr_rf_e = RandomForestRegressor(max_depth=max_depth, 
                                      random_state=2,
                                      n_estimators=30)                                                           
    regr_rf_e.fit(train[features],
                  train['mcEnergy'])
    
    print("Random Forest trained!")    
    print("Training Random Forest Regressor for Disp Reconstruction...")
    
    regr_rf_disp = RandomForestRegressor(max_depth=max_depth,
                                         random_state=2,
                                         n_estimators=30)                                                           
    regr_rf_disp.fit(train[features],
                     train['disp'])
    
    print("Random Forest trained!")
    print("Done!")
    return regr_rf_e, regr_rf_disp

def trainRFsep(train,features):
    
    """Trains a Random Forest classifier for Gamma/Hadron separation.
    Returns the trained RF.

    Parameters:
    -----------
    train: pandas DataFrame
    data set for training the RF

    features: list of strings
    List of features to train the RF

    Return:
    -------
    RandomForestClassifier: clf
    """
    print("Given features: ",features)
    print("Number of events for training: ",train.shape[0])
    print("Training Random Forest Classifier for",
    "Gamma/Hadron separation...")
    
    clf = RandomForestClassifier(max_depth = 50,
                             n_jobs=10,
                             random_state=4,
                             n_estimators=30)
    
    clf.fit(train[features],
            train['hadroness'])
    print("Random Forest trained!")
    print("Done!")
    return clf 


def buildModels(filegammas,fileprotons,features,
                SaveModels=True,path_models="",
                EnergyCut=-1,IntensityCut=60,rCut=0.94):
    """Uses MC data to train Random Forests for Energy and Disp
    reconstruction and G/H separation. Returns 3 trained RF.

    Parameters:
    -----------
    filegammas: string
    Name of the file with MC gamma events
    
    fileprotons: string
    Name of the file with MC proton events

    features: list of strings
    Features for traininf the RF

    EnergyCut: float 
    Cut in energy for gamma/hadron separation
    
    IntensityCut: float
    Cut in intensity of the showers for training RF. Default is 60 phe

    rCut: float
    Cut in distance from c.o.g of hillas ellipse to camera center, to avoid images truncated
    in the border. Default is 80% of camera radius.

    SaveModels: boolean
    Save the trained RF in a file to use them anytime. 
    
    path_models: string
    path to store the trained RF

    Returns:
    --------
    RandomForestRegressor: RFreg_Energy
    RandomForestRegressor: RFreg_Disp
    RandomForestClassifier: RFcls_GH
    """
    
    df_gamma = pd.read_hdf(filegammas,
                           key='gamma_events')
    df_proton = pd.read_hdf(fileprotons,
                            key='proton_events')

    #Apply cuts in intensity and r

    df_gamma = df_gamma[abs(df_gamma['r'])<rCut]
    df_proton = df_proton[abs(df_proton['r'])<rCut]

    #Cut showers with low intensity
    df_gamma = df_gamma[abs(df_gamma['intensity'])>np.log10(IntensityCut)]
    df_proton = df_proton[abs(df_proton['intensity'])>np.log10(IntensityCut)]
    
    #Train regressors for energy and disp reconstruction, only with gammas
    
    RFreg_Energy, RFreg_Disp = trainRFreco(df_gamma,
                                           features)

    #Train classifier for gamma/hadron separation. We need to use half
    #of the gammas for training regressors and have Erec and Disp rec
    #for training the classifier.

    train, testg = split_traintest(df_gamma,
                                   0.5)
    test = testg.append(df_proton,
                        ignore_index=True)

    tempRFreg_Energy, tempRFreg_Disp = trainRFreco(train,features)
    
    #Apply the regressors to the test set

    test['Erec'] = tempRFreg_Energy.predict(test[features])
    test['Disprec'] = tempRFreg_Disp.predict(test[features])
    
    #Apply cut in reconstructed energy. New train set is the previous
    #test with energy and disp reconstructed.
    
    train = test[test['mcEnergy']>EnergyCut]
    
    del tempRFreg_Energy, tempRFreg_Disp
    
    #Add Erec and Disprec to features.
    features_sep = features
    features_sep.append('Erec')
    features_sep.append('Disprec')
    
    #Train the Classifier

    RFcls_GH = trainRFsep(train,
                          features_sep) 
    
    if SaveModels==True:
        fileE = path_models+"RFreg_Energy.sav"
        fileD = path_models+"RFreg_Disp.sav"
        fileH = path_models+"RFcls_GH.sav"
        joblib.dump(RFreg_Energy, fileE)
        joblib.dump(RFreg_Disp, fileD)
        joblib.dump(RFcls_GH, fileH)
        
    features.remove('Erec')
    features.remove('Disprec')
    return RFreg_Energy,RFreg_Disp,RFcls_GH

def ApplyModels(dl1,dl2,features,RFcls_GH,RFreg_Energy,RFreg_Disp):
    """Apply previously trained Random Forests to a set of data
    depending on a set of features.

    Parameters:
    -----------
    data: Pandas DataFrame
    
    features: list

    RFcls_GH: Random Forest Classifier
    RF for Gamma/Hadron separation

    RFreg_Energy: Random Forest Regressor
    RF for Energy reconstruction

    RFreg_Disp: Random Forest Regressor
    RF for Disp reconstruction

    """
    dl2 = dl1
    #Reconstruction of Energy and Disp distance
    dl2['Erec'] = RFreg_Energy.predict(dl2[features])
    dl2['Disprec'] = RFreg_Disp.predict(dl2[features])
   
    #Construction of Source position in camera coordinates from Disp distance.
    #WARNING: For not it only works fine for POINT SOURCE events
    dl2['SrcXrec'],dl2['SrcYrec'] = utils.Disp_to_Pos(dl2['Disprec'],
                                                                  dl2['x'],
                                                                  dl2['y'],
                                                                  dl2['psi'])
    
    features.append('Erec')
    features.append('Disprec')
    dl2['Hadrorec'] = RFcls_GH.predict(dl2[features])
    

