"""Module with functions for Energy and disp_norm reconstruction and G/H
separation. There are functions for raining random forest and for
applying them to data. The RF can be saved into a file for later use.

Usage:

"import dl1_to_dl2"
"""
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import os
from . import utils
from astropy.utils import deprecated
from ..io import read_configuration_file


__all__ = [
    'train_energy',
    'train_disp_norm',
    'train_disp_sign',
    'train_disp_vector',
    'train_reco',
    'train_sep',
    'build_models',
    'apply_models',
    'filter_events',
]


# Standard models configurations - to be moved later in a default configuration file

random_forest_regressor_args = {'max_depth': 50,
                                'min_samples_leaf': 5,
                                'n_jobs': 4,
                                'n_estimators': 100,
                                'bootstrap': True,
                                'criterion': 'mse',
                                'max_features': 'auto',
                                'max_leaf_nodes': None,
                                'min_impurity_decrease': 0.0,
                                'min_impurity_split': None,
                                'min_samples_split': 2,
                                'min_weight_fraction_leaf': 0.0,
                                'oob_score': False,
                                'random_state': 42,
                                'verbose': 0,
                                'warm_start': False,
                                }


random_forest_classifier_args = {'max_depth': 50,
                                 'min_samples_leaf': 2,
                                 'n_jobs': 4,
                                 'n_estimators': 100,
                                 'criterion': 'gini',
                                 'min_samples_split': 2,
                                 'min_weight_fraction_leaf': 0.,
                                 'max_features': 'auto',
                                 'max_leaf_nodes': None,
                                 'min_impurity_decrease': 0.0,
                                 'min_impurity_split': None,
                                 'bootstrap': True,
                                 'oob_score': False,
                                 'random_state': 42,
                                 'verbose': 0.,
                                 'warm_start': False,
                                 'class_weight': None,
                                 }



@deprecated('31/10/2019', message='Will be removed in a future release')
def split_traintest(data, proportion, random_state=42):
    """
    Split a dataset in "train" and "test" sets.
    Actually using `sklearn.model_selection.train_test_split`

    Parameters:
    -----------
    data: pandas DataFrame
    proportion: float
    Percentage of the total dataset that will be part of the train set.

    Returns:
    --------
    train, test - `pandas.DataFrame`
    """
    train, test = train_test_split(data, train_size=proportion, random_state=random_state)
    return train, test


def train_energy(train,
                 features,
                 model=RandomForestRegressor,
                 regression_args=random_forest_regressor_args,
                 config_file=None):
    """
    Train a model for the regression of the energy

    Parameters
    ----------
    train: `pandas.DataFrame`
    features: list of strings, features to train the model
    model: `scikit-learn` model with a `fit` method. By default `sklearn.ensemble.RandomForestRegressor`
    model_args: dictionnary, arguments for the model
    config_file: str - path to a configuration file. If given, overwrite `model_args`.

    Returns
    -------
    The trained model
    """
    if config_file is not None:
        config = read_configuration_file(config_file)
        regression_args = config['random_forest_regressor_args']

    print("Given features: ", features)
    print("Number of events for training: ", train.shape[0])
    print("Training Random Forest Regressor for Energy Reconstruction...")

    reg = model(**regression_args)
    reg.fit(train[features],
                  train['mc_energy'])

    print("Model {} trained!".format(model))
    return reg


def train_disp_vector(train, features,
        model=RandomForestRegressor,
        regression_args=random_forest_regressor_args,
        config_file=None,
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
    regression_args_args: dictionnary, arguments for the model
    config_file: str - path to a configuration file. If given, overwrites `model_args`.

    Returns
    -------
    The trained model
    """
    if config_file is not None:
        config = read_configuration_file(config_file)
        regression_args = config['random_forest_regressor_args']

    print("Given features: ", features)
    print("Number of events for training: ", train.shape[0])
    print("Training model {} for disp vector regression".format(model))

    reg = model(**regression_args)
    x = train[features]
    y = np.transpose([train[f] for f in predict_features])
    reg.fit(x, y)

    print("Model {} trained!".format(model))

    return reg


def train_disp_norm(train, features,
        model=RandomForestRegressor,
        regression_args=random_forest_regressor_args,
        config_file=None,
        predict_feature='disp_norm'):
    """
    Train a model for the regression of the disp_norm norm

    Parameters
    ----------
    train: `pandas.DataFrame`
    features: list of strings, features to train the model
    model: `scikit-learn` model with a `fit` method.
    By default `sklearn.ensemble.RandomForestRegressor`
    regression_args: dictionnary, arguments for the model
    config_file: str - path to a configuration file. If given, overwrites `regression_args`.

    Returns
    -------
    The trained model
    """
    if config_file is not None:
        config = read_configuration_file(config_file)
        regression_args = config['random_forest_regressor_args']

    print("Given features: ", features)
    print("Number of events for training: ", train.shape[0])
    print("Training model {} for disp norm regression".format(model))

    reg = model(**regression_args)
    x = train[features]
    y = np.transpose(train[predict_feature])
    reg.fit(x, y)

    print("Model {} trained!".format(model))

    return reg


def train_disp_sign(train, features,
        model=RandomForestClassifier,
        classification_args=random_forest_classifier_args,
        config_file=None,
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
    config_file: str - path to a configuration file. If given, overwrite `model_args`.

    Returns
    -------
    The trained model
    """
    if config_file is not None:
        config = read_configuration_file(config_file)
        classification_args = config['random_forest_classifier_args']

    print("Given features: ", features)
    print("Number of events for training: ", train.shape[0])
    print("Training model {} for disp sign regression".format(model))

    reg = model(**classification_args)
    x = train[features]
    y = np.transpose(train[predict_feature])
    reg.fit(x, y)

    print("Model {} trained!".format(model))

    return reg



def train_reco(train, features, regression_args=random_forest_regressor_args, config_file=None):
    """
    Trains two Random Forest regressors for Energy and disp_norm
    reconstruction respectively. Returns the trained RF.

    Parameters:
    -----------
    train: `pandas.DataFrame`
    data set for training the RF
    features: list of strings
    List of features to train the RF
    regression_args: dictionnary
    config_file: str - path to a configuration file. If given, overwrite `regression_args`.

    Returns:
    --------
    RandomForestRegressor: reg_energy
    RandomForestRegressor: reg_disp
    """
    if config_file is not None:
        config = read_configuration_file(config_file)
        regression_args = config['random_forest_regressor_args']


    print("Given features: ", features)
    print("Number of events for training: ", train.shape[0])
    print("Training Random Forest Regressor for Energy Reconstruction...")

    reg_energy = RandomForestRegressor(**regression_args)
    reg_energy.fit(train[features],
                  train['mc_energy'])

    print("Random Forest trained!")
    print("Training Random Forest Regressor for disp_norm Reconstruction...")

    reg_disp = RandomForestRegressor(**regression_args)
    reg_disp.fit(train[features],
                     train['disp_norm'])

    print("Random Forest trained!")
    print("Done!")
    return reg_energy, reg_disp


def train_sep(train, features, classification_args=random_forest_classifier_args, config_file=None):

    """Trains a Random Forest classifier for Gamma/Hadron separation.
    Returns the trained RF.

    Parameters:
    -----------
    train: `pandas.DataFrame`
    data set for training the RF
    features: list of strings
    List of features to train the RF
    classification_args: dictionnary
    config_file: str - path to a configuration file. If given, overwrite `classification_args`.

    Return:
    -------
    `RandomForestClassifier`
    """
    if config_file is not None:
        config = read_configuration_file(config_file)
        classification_args = config['random_forest_classifier_args']

    print("Given features: ", features)
    print("Number of events for training: ", train.shape[0])
    print("Training Random Forest Classifier for",
    "Gamma/Hadron separation...")

    clf = RandomForestClassifier(**classification_args)

    clf.fit(train[features],
            train['mc_type'])
    print("Random Forest trained!")
    return clf


def build_models(filegammas, fileprotons, features,
                save_models=True, path_models="./",
                energy_min=-1, intensity_min=np.log10(60), leakage_cut=0.2,
                r_min=0.15,
                regression_args=random_forest_regressor_args,
                classification_args=random_forest_classifier_args,
                config_file=None):
    """Uses MC data to train Random Forests for Energy and disp_norm
    reconstruction and G/H separation. Returns 3 trained RF.

    Parameters:
    -----------
    filegammas: string
    Name of the file with MC gamma events

    fileprotons: string
    Name of the file with MC proton events

    features: list of strings
    Features for training the RF

    energy_min: float
    Cut in energy for gamma/hadron separation

    intensity_min: float
    Cut in intensity of the showers for training RF. Default is 60 phe

    r_min: float
    Cut in distance from c.o.g of hillas ellipse to camera center, to avoid images truncated
    in the border. Default is 80% of camera radius.

    save_models: boolean
    Save the trained RF in a file to use them anytime.

    path_models: string
    path to store the trained RF

    regression_args: dictionnary

    classification_args: dictionnary

    config_file: str - path to a configuration file. If given, overwrite `regression_args`.

    Returns:
    --------
    (regressor_energy, regressor_disp, classifier_gh)
    regressor_energy: `RandomForestRegressor`
    regressor_disp: `RandomForestRegressor`
    classifier_gh: `RandomForestClassifier`
    """
    if config_file is not None:
        config = read_configuration_file(config_file)
        regression_args = config['random_forest_regressor_args']
        classification_args = config['random_forest_classifier_args']

    df_gamma = pd.read_hdf(filegammas, key='events/LSTCam')
    df_proton = pd.read_hdf(fileprotons, key='events/LSTCam')

    df_gamma = filter_events(df_gamma,
                             leakage_cut=leakage_cut,
                             intensity_min=intensity_min,
                             r_min=r_min)
    df_proton = filter_events(df_proton,
                              leakage_cut=leakage_cut,
                              intensity_min=intensity_min,
                              r_min=r_min)

    #Train regressors for energy and disp_norm reconstruction, only with gammas

    reg_energy = train_energy(df_gamma, features,
                            regression_args=regression_args
                          )
    reg_disp_vector = train_disp_vector(df_gamma, features,
                            regression_args=regression_args
                          )

    #Train classifier for gamma/hadron separation.

    train, testg = train_test_split(df_gamma, test_size=0.2)
    test = testg.append(df_proton, ignore_index=True)

    temp_reg_energy = train_energy(train, features,
                            regression_args=regression_args
                          )
    temp_reg_disp_vector = train_disp_vector(train, features,
                            regression_args=regression_args
                          )

    #Apply the regressors to the test set

    test['e_rec'] = temp_reg_energy.predict(test[features])
    disp_vector = temp_reg_disp_vector.predict(test[features])
    test['disp_dx_rec'] = disp_vector[:,0]
    test['disp_dy_rec'] = disp_vector[:,1]

    #Apply cut in reconstructed energy. New train set is the previous
    #test with energy and disp_norm reconstructed.

    train = test[test['e_rec'] > energy_min]

    del temp_reg_energy, temp_reg_disp_vector

    #Add e_rec and disp_rec to features.
    features_sep = features.copy()
    features_sep.append('e_rec')
    features_sep.append('disp_dx_rec')
    features_sep.append('disp_dy_rec')

    #Train the Classifier

    cls_gh = train_sep(train, features_sep, classification_args=classification_args)

    if save_models:
        os.makedirs(path_models, exist_ok=True)
        file_reg_energy = path_models + "/reg_energy.sav"
        file_reg_disp_vector = path_models + "/reg_disp_vector.sav"
        file_cls_gh = path_models + "/cls_gh.sav"
        joblib.dump(reg_energy, file_reg_energy)
        joblib.dump(reg_disp_vector, file_reg_disp_vector)
        joblib.dump(cls_gh, file_cls_gh)

    return reg_energy, reg_disp_vector, cls_gh


def apply_models(dl1, features, classifier, reg_energy, reg_disp_vector):
    """Apply previously trained Random Forests to a set of data
    depending on a set of features.

    Parameters:
    -----------
    data: Pandas DataFrame

    features: list

    classifier: Random Forest Classifier
    RF for Gamma/Hadron separation

    reg_energy: Random Forest Regressor
    RF for Energy reconstruction

    reg_disp: Random Forest Regressor
    RF for disp_norm reconstruction

    """

    features_ = list(features)
    dl2 = dl1.copy()
    #Reconstruction of Energy and disp_norm distance
    dl2['e_rec'] = reg_energy.predict(dl2[features_])
    disp_vector = reg_disp_vector.predict(dl2[features_])
    dl2['disp_dx_rec'] = disp_vector[:,0]
    dl2['disp_dy_rec'] = disp_vector[:,1]

    #Construction of Source position in camera coordinates from disp_norm distance.
    #WARNING: For not it only works fine for POINT SOURCE events


    dl2['src_x_rec'], dl2['src_y_rec'] = utils.disp_to_pos(dl2.disp_dx_rec,
                                                           dl2.disp_dy_rec,
                                                           dl2.x,
                                                           dl2.y,
                                                           )

    features_.append('e_rec')
    features_.append('disp_dx_rec')
    features_.append('disp_dy_rec')
    dl2['reco_type'] = classifier.predict(dl2[features_]).astype(int)
    probs = classifier.predict_proba(dl2[features_])[0:,0]
    dl2['gammaness'] = probs
    return dl2


def filter_events(data, leakage_cut = 1.0, intensity_min = 10, r_min = 0.15):
    """
    Filter events based on extracted features.

    Parameters
    ----------
    data: `pandas.DataFrame`

    Returns
    -------
    `pandas.DataFrame`
    """

    filter = (data['leakage'] < leakage_cut) & (data['intensity'] > intensity_min) & (data['r'] > r_min)
    return data[filter]
