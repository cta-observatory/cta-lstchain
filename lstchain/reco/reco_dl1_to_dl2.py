"""Module with functions for Energy and disp_norm reconstruction and G/H
separation. There are functions for raining random forest and for
applying them to data. The RF can be saved into a file for later use.

Usage:

"import reco_dl1_to_dl2"
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import os

from . import utils

def split_traintest(data, proportion, random_state=42):
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
    # data['is_train'] = np.random.uniform(0, 1, len(data)) <= proportion
    # train = data[data['is_train']==True]
    # test = data[data['is_train']==False]
    train, test = train_test_split(data, train_size=proportion, random_state=random_state)
    return train, test

def train_energy(train,
                 features,
                 model=RandomForestRegressor,
                 model_args={'max_depth': 50,
                            'min_samples_leaf': 50,
                            'n_jobs': 4,
                            'n_estimators': 50}):
    """
    Train a model for the regression of the energy

    Parameters
    ----------
    train: `pandas.DataFrame`
    features: list of strings, features to train the model
    model: `scikit-learn` model with a `fit` method. By default `sklearn.ensemble.RandomForestRegressor`
    model_args: dictionnary, arguments for the model

    Returns
    -------
    The trained model
    """

    print("Given features: ", features)
    print("Number of events for training: ", train.shape[0])
    print("Training Random Forest Regressor for Energy Reconstruction...")

    reg = model(**model_args)
    reg.fit(train[features],
                  train['mc_energy'])

    print("Model {} trained!".format(model))
    return reg


def train_disp_vector(train, features,
        model=RandomForestRegressor,
        model_args={'max_depth': 2,
                    'min_samples_leaf': 50,
                    'n_jobs': 4,
                    'n_estimators': 10},
        predict_features=['disp_dx', 'disp_dy']):
    """
    Train a model for the regression of the disp_norm vector coordinates dx,dy.
    Therefore, the model must be able to be applied on a vector of features.

    Parameters
    ----------
    train: `pandas.DataFrame`
    features: list of strings, features to train the model
    model: `scikit-learn` model with a `fit` method that can be applied to a vector of features.
    By default `sklearn.ensemble.RandomForestRegressor`
    model_args: dictionnary, arguments for the model

    Returns
    -------
    The trained model
    """

    print("Given features: ", features)
    print("Number of events for training: ", train.shape[0])
    print("Training mdoel {} for disp_norm vector regression".format(model))

    reg = model(**model_args)
    x = train[features]
    y = np.transpose([train[f] for f in predict_features])
    reg.fit(x, y)

    print("Model {} trained!".format(model))

    return reg


def train_disp_norm(train, features,
        model=RandomForestRegressor,
        model_args={'max_depth': 2,
                    'min_samples_leaf': 50,
                    'n_jobs': 4,
                    'n_estimators': 10},
        predict_feature='disp_norm'):
    """
    Train a model for the regression of the disp_norm norm

    Parameters
    ----------
    train: `pandas.DataFrame`
    features: list of strings, features to train the model
    model: `scikit-learn` model with a `fit` method.
    By default `sklearn.ensemble.RandomForestRegressor`
    model_args: dictionnary, arguments for the model

    Returns
    -------
    The trained model
    """

    print("Given features: ", features)
    print("Number of events for training: ", train.shape[0])
    print("Training mdoel {} for disp_norm vector regression".format(model))

    reg = model(**model_args)
    x = train[features]
    y = np.transpose(train[predict_feature])
    reg.fit(x, y)

    print("Model {} trained!".format(model))

    return reg


def train_disp_sign(train, features,
        model=RandomForestClassifier,
        model_args={'max_depth': 2,
                    'min_samples_leaf': 50,
                    'n_jobs': 4,
                    'n_estimators': 10},
        predict_feature='disp_sign'):
    """
    Train a model for the classification of the disp_norm sign

    Parameters
    ----------
    train: `pandas.DataFrame`
    features: list of strings, features to train the model
    model: `scikit-learn` model with a `fit` method.
    By default `sklearn.ensemble.RandomForestClassifier`
    model_args: dictionnary, arguments for the model

    Returns
    -------
    The trained model
    """

    print("Given features: ", features)
    print("Number of events for training: ", train.shape[0])
    print("Training mdoel {} for disp_norm vector regression".format(model))

    reg = model(**model_args)
    x = train[features]
    y = np.transpose(train[predict_feature])
    reg.fit(x, y)

    print("Model {} trained!".format(model))

    return reg



def trainRFreco(train, features):
    """
    Trains two Random Forest regressors for Energy and disp_norm
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
                                      min_samples_leaf=50,
                                      n_jobs=4,
                                      n_estimators=50)
    regr_rf_e.fit(train[features],
                  train['mc_energy'])
    
    print("Random Forest trained!")    
    print("Training Random Forest Regressor for disp_norm Reconstruction...")
    
    regr_rf_disp = RandomForestRegressor(max_depth=max_depth,
                                         min_samples_leaf=50,
                                         n_jobs=4,
                                         n_estimators=50)    
    regr_rf_disp.fit(train[features],
                     train['disp_norm'])
    
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
                                 n_jobs=4,
                                 min_samples_leaf=50,
                                 n_estimators=100)
    
    clf.fit(train[features],
            train['hadroness'])
    print("Random Forest trained!")
    print("Done!")
    return clf 


def buildModels(filegammas, fileprotons, features,
                SaveModels=True, path_models="./",
                EnergyCut=-1, IntensityCut=60, rCut=0.94):
    """Uses MC data to train Random Forests for Energy and disp_norm
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
    if not os.path.exists(path_models):
        os.mkdir(path_models)

    features_=list(features)

    df_gamma = pd.read_hdf(filegammas)
    df_proton = pd.read_hdf(fileprotons)

    #Apply cuts in intensity and r

    df_gamma = df_gamma[abs(df_gamma['r'])<rCut]
    df_proton = df_proton[abs(df_proton['r'])<rCut]

    #Cut showers with low intensity
    df_gamma = df_gamma[abs(df_gamma['intensity']) > np.log10(IntensityCut)]
    df_proton = df_proton[abs(df_proton['intensity']) > np.log10(IntensityCut)]
    
    #Train regressors for energy and disp_norm reconstruction, only with gammas
    
    RFreg_Energy, RFreg_Disp = trainRFreco(df_gamma, features)

    #Train classifier for gamma/hadron separation. We need to use half
    #of the gammas for training regressors and have e_rec and disp_norm rec
    #for training the classifier.

    train, testg = split_traintest(df_gamma, 0.5)
    test = testg.append(df_proton, ignore_index=True)

    tempRFreg_Energy, tempRFreg_Disp = trainRFreco(train, features_)
    
    #Apply the regressors to the test set

    test['e_rec'] = tempRFreg_Energy.predict(test[features_])
    test['disp_rec'] = tempRFreg_Disp.predict(test[features_])
    
    #Apply cut in reconstructed energy. New train set is the previous
    #test with energy and disp_norm reconstructed.
    
    train = test[test['mc_energy']>EnergyCut]
    
    del tempRFreg_Energy, tempRFreg_Disp
    
    #Add e_rec and disp_rec to features.
    features_sep = features_
    features_sep.append('e_rec')
    features_sep.append('disp_rec')
    
    #Train the Classifier

    RFcls_GH = trainRFsep(train,
                          features_sep) 
    
    if SaveModels:
        fileE = path_models + "/RFreg_Energy.sav"
        fileD = path_models + "/RFreg_Disp.sav"
        fileH = path_models + "/RFcls_GH.sav"
        joblib.dump(RFreg_Energy, fileE)
        joblib.dump(RFreg_Disp, fileD)
        joblib.dump(RFcls_GH, fileH)

    return RFreg_Energy, RFreg_Disp, RFcls_GH


def ApplyModels(dl1, features, RFcls_GH, RFreg_Energy, RFreg_Disp):
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
    RF for disp_norm reconstruction

    """
    
    features_ = list(features)
    dl2 = dl1.copy()
    #Reconstruction of Energy and disp_norm distance
    dl2['e_rec'] = RFreg_Energy.predict(dl2[features_])
    dl2['disp_rec'] = RFreg_Disp.predict(dl2[features_])
   
    #Construction of Source position in camera coordinates from disp_norm distance.
    #WARNING: For not it only works fine for POINT SOURCE events

    disp_norm = dl2['disp_rec']
    disp_angle = dl2['psi']
    disp_sign = utils.source_side(0, dl2['x'])
    disp_dx, disp_dy = utils.disp_vector(disp_norm, disp_angle, disp_sign)

    dl2['src_x_rec'], dl2['src_y_rec'] = utils.disp_to_pos(disp_dx,
                                                           disp_dy,
                                                           dl2['x'],
                                                           dl2['y'],
                                                           )
    
    features_.append('e_rec')
    features_.append('disp_rec')
    dl2['hadro_rec'] = RFcls_GH.predict(dl2[features_]).astype(int)

    return dl2

