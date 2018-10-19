import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve


def split_traintest(data,proportion):
    """
    Split a dataset in "train" and "test" sets.

    Parameters:
    data: pandas DataFrame

    proportion: float
    Percentage of the total dataset that will be part of the train set.
    
    """
    data['is_train'] = np.random.uniform(0,1,len(data))<= proportion
    train = data[(data['is_train']==True)]
    test = data[(data['is_train']==False)]
    return train,test

def trainRFreco(train,features):
    """
    Trains two Random Forest regressors for Energy and Disp reconstruction respectively.
    Returns the trained RF.

    Parameters:
    train: pandas DataFrame
    data set for training the RF

    features: list of strings
    List of features to train the RF

    """
    max_depth = 50
    regr_rf_e = RandomForestRegressor(max_depth=max_depth, random_state=2,n_estimators=100)                                                           
    regr_rf_e.fit(train[features], train['mcEnergy'])
    #erec = regr_rf_e.predict(test[features])

    #Reconstruct Disp
    regr_rf_disp = RandomForestRegressor(max_depth=max_depth, random_state=2,n_estimators=100)                                                           
    regr_rf_disp.fit(train[features], train['disp'])
    #disprec = regr_rf_disp.predict(test[features])
        
    return regr_rf_e, regr_rf_disp

def trainRFsep(train,features):
    
    """
    Trains a Random Forest classifier for Gamma/Hadron separation.
    Returns the trained RF.

    Parameters:
    train: pandas DataFrame
    data set for training the RF

    features: list of strings
    List of features to train the RF

    """
    clf = RandomForestClassifier(max_depth = 50,
                             n_jobs=10,
                             random_state=4,
                             n_estimators=500)
    
    clf.fit(train[features],train['hadroness'])
    return clf 

def buildModels(filegammas,fileprotons,features,EnergyCut=-1,IntensityCut=60,rCut=0.94):
    """
    Uses MC data to train Random Forests for Energy and Disp reconstruction and G/H separation.
    Returns 3 trained RF.

    Parameters:
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
    
    """
    
    df_gamma = pd.read_hdf(filegammas,key='gamma_events')
    df_proton = pd.read_hdf(fileprotons,key='proton_events')

    #Apply cuts in intensity and r

    df_gamma = df_gamma[abs(df_gamma['r'])<0.94]
    df_proton = df_proton[abs(df_proton['r'])<0.94]

    #Cut showers with low intensity
    df_gamma = df_gamma[abs(df_gamma['intensity'])>60]
    df_proton = df_proton[abs(df_proton['intensity'])>60]
    
    #Train regressors for energy and disp reconstruction, only with gammas
    
    RFreg_Energy, RFreg_Disp = trainRFreco(df_gamma,features)

    #Train classifier for gamma/hadron separation. We need to use half of the gammas for
    #training regressors and have Erec and Disp rec for training the classifier

    train, testg = split_traintest(df_gamma,0.5)
    test = testg.append(df_proton,ignore_index=True)

    tempRFreg_Energy, tempRFreg_Disp = trainRFreco(traing,features)
    
    #Apply the regressors to the test set

    test['Erec'] = tempRFreg_Energy.predict(test[features])
    test['Disprec'] = tempRFreg_Disp(test[features])
    
    #Apply cut in reconstructed energy. New train set is the previous test with energy and disp reconstructed.
    train = test[test['mcEnergy']>EnergyCut]
    
    #Add Erec and Disprec to features.

    features.append('Erec')
    features.append('Disprec')

    #Train the Classifier

    RFcls_GH = trainRFsep(train,features) 
    
    return RFreg_Energy,RFreg_Disp,RFcls_GH
