"""Module with functions for Energy and disp_norm reconstruction and G/H
separation. There are functions for raining random forest and for
applying them to data. The RF can be saved into a file for later use.

Usage:

"import dl1_to_dl2"
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os
from astropy.time import Time
from astropy.coordinates import SkyCoord
from . import utils
from . import disp
from ..io import standard_config, replace_config
import astropy.units as u
from ..io.io import dl1_params_lstcam_key, dl1_params_src_dep_lstcam_key
from ctapipe.image.hillas import camera_to_shower_coordinates


__all__ = [
    'train_energy',
    'train_disp_norm',
    'train_disp_sign',
    'train_disp_vector',
    'train_reco',
    'train_sep',
    'build_models',
    'apply_models',
    'get_source_dependent_parameters',
    'get_expected_source_pos'
]



def train_energy(train, custom_config={}):
    """
    Train a Random Forest Regressor for the regression of the energy
    TODO: introduce the possibility to use another model

    Parameters
    ----------
    train: `pandas.DataFrame`
    config: dictionnary containing configuration

    Returns
    -------
    The trained model
    """

    config = replace_config(standard_config, custom_config)
    regression_args = config['random_forest_regressor_args'] 
    features = config['regression_features']
    model = RandomForestRegressor    

    print("Given features: ", features)
    print("Number of events for training: ", train.shape[0])
    print("Training Random Forest Regressor for Energy Reconstruction...")

    reg = model(**regression_args)
    reg.fit(train[features],
                  train['log_mc_energy'])

    print("Model {} trained!".format(model))
    return reg


def train_disp_vector(train, custom_config={}, predict_features=['disp_dx', 'disp_dy']):
    """
    Train a model (Random Forest Regressor) for the regression of the disp_norm vector coordinates dx,dy.
    Therefore, the model must be able to be applied on a vector of features.
    TODO: introduce the possibility to use another model

    Parameters
    ----------
    train: `pandas.DataFrame`
    config: dictionnary containing configuration

    Returns
    -------
    The trained model
    """

    config = replace_config(standard_config, custom_config)
    regression_args = config['random_forest_regressor_args']
    features = config['regression_features']
    model = RandomForestRegressor

    print("Given features: ", features)
    print("Number of events for training: ", train.shape[0])
    print("Training model {} for disp vector regression".format(model))

    reg = model(**regression_args)
    x = train[features]
    y = np.transpose([train[f] for f in predict_features])
    reg.fit(x, y)

    print("Model {} trained!".format(model))

    return reg


def train_disp_norm(train, custom_config={}, predict_feature='disp_norm'):
    """
    Train a model for the regression of the disp_norm norm

    Parameters
    ----------
    train: `pandas.DataFrame`
    config: dictionnary containing configuration

    Returns
    -------
    The trained model
    """

    config = replace_config(standard_config, custom_config)
    regression_args = config['random_forest_regressor_args']
    features = config['regression_features']
    model = RandomForestRegressor

    print("Given features: ", features)
    print("Number of events for training: ", train.shape[0])
    print("Training model {} for disp norm regression".format(model))

    reg = model(**regression_args)
    x = train[features]
    y = np.transpose(train[predict_feature])
    reg.fit(x, y)

    print("Model {} trained!".format(model))

    return reg


def train_disp_sign(train, custom_config={}, predict_feature='disp_sign'):
    """
    Train a model for the classification of the disp_norm sign

    Parameters
    ----------
    train: `pandas.DataFrame`
    config: dictionnary containing configuration

    Returns
    -------
    The trained model
    """

    config = replace_config(standard_config, custom_config)
    classification_args = config['random_forest_classifier_args']
    features = config["classification_features"]
    model = RandomForestClassifier

    print("Given features: ", features)
    print("Number of events for training: ", train.shape[0])
    print("Training model {} for disp sign regression".format(model))

    reg = model(**classification_args)
    x = train[features]
    y = np.transpose(train[predict_feature])
    reg.fit(x, y)

    print("Model {} trained!".format(model))

    return reg



def train_reco(train, custom_config={}):
    """
    Trains two Random Forest regressors for Energy and disp_norm
    reconstruction respectively. Returns the trained RF.

    Parameters:
    -----------
    train: `pandas.DataFrame`
    config: dictionnary containing configuration

    Returns:
    --------
    RandomForestRegressor: reg_energy
    RandomForestRegressor: reg_disp
    """

    config = replace_config(standard_config, custom_config)
    regression_args = config['random_forest_regressor_args']
    features = config['regression_features']
    model = RandomForestRegressor

    print("Given features: ", features)
    print("Number of events for training: ", train.shape[0])
    print("Training Random Forest Regressor for Energy Reconstruction...")

    reg_energy = model(**regression_args)
    reg_energy.fit(train[features],
                  train['log_mc_energy'])

    print("Random Forest trained!")
    print("Training Random Forest Regressor for disp_norm Reconstruction...")

    reg_disp = RandomForestRegressor(**regression_args)
    reg_disp.fit(train[features],
                     train['disp_norm'])

    print("Random Forest trained!")
    print("Done!")
    return reg_energy, reg_disp


def train_sep(train, custom_config={}):

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

    config = replace_config(standard_config, custom_config)
    classification_args = config['random_forest_classifier_args']
    features = config["classification_features"]
    model = RandomForestClassifier


    print("Given features: ", features)
    print("Number of events for training: ", train.shape[0])
    print("Training Random Forest Classifier for",
    "Gamma/Hadron separation...")

    clf = model(**classification_args)

    clf.fit(train[features],
            train['mc_type'])
    print("Random Forest trained!")
    return clf


def build_models(filegammas, fileprotons,
                 save_models=True, path_models="./",
                 energy_min=-np.inf,
                 custom_config={},
                 test_size=0.2,
                 ):
    """Uses MC data to train Random Forests for Energy and disp_norm
    reconstruction and G/H separation. Returns 3 trained RF.
    The config in config_file superseeds the one passed in argument.

    Parameters:
    -----------
    filegammas: string
    Name of the file with MC gamma events

    fileprotons: string
    Name of the file with MC proton events

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

    config = replace_config(standard_config, custom_config)
    events_filters = config["events_filters"]

    # Adding a filter on mc_type just for training
    events_filters['mc_type'] = [-9000, np.inf]

    df_gamma = pd.read_hdf(filegammas, key=dl1_params_lstcam_key)
    df_proton = pd.read_hdf(fileprotons, key=dl1_params_lstcam_key)

    if config['source_dependent']:
        src_dep_df_gamma = pd.read_hdf(filegammas, key=dl1_params_src_dep_lstcam_key)
        src_dep_df_gamma.columns = pd.MultiIndex.from_tuples([tuple(col[1:-1].replace('\'', '').replace(' ','').split(",")) for col in src_dep_df_gamma.columns])
        df_gamma = pd.concat([df_gamma, src_dep_df_gamma['on']], axis=1)

        src_dep_df_proton = pd.read_hdf(fileprotons, key=dl1_params_src_dep_lstcam_key)
        src_dep_df_proton.columns = pd.MultiIndex.from_tuples([tuple(col[1:-1].replace('\'', '').replace(' ','').split(",")) for col in src_dep_df_proton.columns])
        df_proton = pd.concat([df_proton, src_dep_df_proton['on']], axis=1)

    df_gamma = utils.filter_events(df_gamma,
                                   filters=events_filters,
                                   finite_params=config['regression_features'] + config['classification_features'],
                                   )

    df_proton = utils.filter_events(df_proton,
                                    filters=events_filters,
                                    finite_params=config['regression_features'] + config['classification_features'],
                                    )


    #Train regressors for energy and disp_norm reconstruction, only with gammas

    reg_energy = train_energy(df_gamma, custom_config=config)

    reg_disp_vector = train_disp_vector(df_gamma, custom_config=config)

    #Train classifier for gamma/hadron separation.

    train, testg = train_test_split(df_gamma, test_size=test_size)
    test = testg.append(df_proton, ignore_index=True)

    temp_reg_energy = train_energy(train, custom_config=config)

    temp_reg_disp_vector = train_disp_vector(train, custom_config=config)

    #Apply the regressors to the test set

    test['log_reco_energy'] = temp_reg_energy.predict(test[config['regression_features']])
    disp_vector = temp_reg_disp_vector.predict(test[config['regression_features']])
    test['reco_disp_dx'] = disp_vector[:, 0]
    test['reco_disp_dy'] = disp_vector[:, 1]

    #Apply cut in reconstructed energy. New train set is the previous
    #test with energy and disp_norm reconstructed.

    train = test[test['log_reco_energy'] > energy_min]

    del temp_reg_energy, temp_reg_disp_vector

    #Train the Classifier

    cls_gh = train_sep(train, custom_config=config)

    if save_models:
        os.makedirs(path_models, exist_ok=True)
        file_reg_energy = path_models + "/reg_energy.sav"
        file_reg_disp_vector = path_models + "/reg_disp_vector.sav"
        file_cls_gh = path_models + "/cls_gh.sav"
        joblib.dump(reg_energy, file_reg_energy)
        joblib.dump(reg_disp_vector, file_reg_disp_vector)
        joblib.dump(cls_gh, file_cls_gh)

    return reg_energy, reg_disp_vector, cls_gh


def apply_models(dl1, classifier, reg_energy, reg_disp_vector, focal_length=28*u.m, custom_config={}):
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

    config = replace_config(standard_config, custom_config)

    dl2 = dl1.copy()

    regression_features = config["regression_features"]
    classification_features = config["classification_features"]
      
    #Reconstruction of Energy and disp_norm distance
    dl2['log_reco_energy'] = reg_energy.predict(dl2[regression_features])
    dl2['reco_energy'] = 10**(dl2['log_reco_energy'])
    disp_vector = reg_disp_vector.predict(dl2[regression_features])
    dl2['reco_disp_dx'] = disp_vector[:, 0]
    dl2['reco_disp_dy'] = disp_vector[:, 1]

    #Construction of Source position in camera coordinates from disp_norm distance.

    dl2['reco_src_x'], dl2['reco_src_y'] = disp.disp_to_pos(dl2.reco_disp_dx,
                                                            dl2.reco_disp_dy,
                                                            dl2.x,
                                                            dl2.y,
                                                            )

    if 'mc_alt_tel' in dl2.columns:
        alt_tel = dl2['mc_alt_tel'].values
        az_tel = dl2['mc_az_tel'].values
    elif 'alt_tel' in dl2.columns:
        alt_tel = dl2['alt_tel'].values
        az_tel = dl2['az_tel'].values
    else:
        alt_tel = - np.pi/2. * np.ones(len(dl2))
        az_tel = - np.pi/2. * np.ones(len(dl2))


    src_pos_reco = utils.reco_source_position_sky(dl2.x.values * u.m,
                                                  dl2.y.values * u.m,
                                                  dl2.reco_disp_dx.values * u.m,
                                                  dl2.reco_disp_dy.values * u.m,
                                                  focal_length,
                                                  alt_tel * u.rad,
                                                  az_tel * u.rad)

    dl2['reco_alt'] = src_pos_reco.alt.rad
    dl2['reco_az'] = src_pos_reco.az.rad

    dl2['reco_type'] = classifier.predict(dl2[classification_features]).astype(int)
    probs = classifier.predict_proba(dl2[classification_features])

    # This check is valid as long as we train on only two classes (gammas and protons)
    if probs.shape[1] > 2:
        raise ValueError("The classifier is predicting more than two classes, "
                         "the predicted probabilty to assign as gammaness is unclear."
                         "Please check training data")

    # gammaness is the prediction probability for the first class (0)
    dl2['gammaness'] = probs[:, 0]

    return dl2



def get_source_dependent_parameters(data, config, focal_length=28*u.m):

    """Get parameters dict for source-dependent analysis .

    Parameters:
    -----------
    data: Pandas DataFrame
    config: dictionnary containing configuration
    
    """

    is_simu = (data['mc_type'] >= 0).all() if 'mc_type' in data.columns else False
    
    if is_simu:
        if (data['mc_type'] == 0).all():
            data_type = 'mc_gamma'
        else:
            data_type = 'mc_proton'
    else:
        data_type = 'real_data'
    
    expected_src_pos_x_m, expected_src_pos_y_m = get_expected_source_pos(data, data_type, config, focal_length=focal_length)

    # ON position
    src_dep_params_dict = {}
    src_dep_params = calc_source_dependent_parameters(data, expected_src_pos_x_m, expected_src_pos_y_m)
    src_dep_params_dict['on'] = src_dep_params

    if not is_simu:
        if config.get('observation_mode')=='wobble':
            for ioff in range(config.get('n_off_wobble')):
                off_angle = 2 * np.pi / (config['n_off_wobble'] + 1) * (ioff + 1)
            
                rotated_expected_src_pos_x_m = expected_src_pos_x_m  * np.cos(off_angle) - expected_src_pos_y_m * np.sin(off_angle)
                rotated_expected_src_pos_y_m = expected_src_pos_x_m  * np.sin(off_angle) + expected_src_pos_y_m * np.cos(off_angle)
                src_dep_params = calc_source_dependent_parameters(data, rotated_expected_src_pos_x_m, rotated_expected_src_pos_y_m)
                src_dep_params['off_angle'] = np.rad2deg(off_angle)
                src_dep_params_dict['off_{:03}'.format(round(np.rad2deg(off_angle)))] = src_dep_params

    return src_dep_params_dict


def calc_source_dependent_parameters(data, expected_src_pos_x_m, expected_src_pos_y_m):
    """Calculate source-dependent parameters with a given source position.

    Parameters:
    -----------
    data: Pandas DataFrame
    expected_src_pos_x_m: float
    expected_src_pos_y_m: float

    """
    src_dep_params = pd.DataFrame(index=data.index)

    src_dep_params['expected_src_x'] = expected_src_pos_x_m
    src_dep_params['expected_src_y'] = expected_src_pos_y_m

    src_dep_params['dist'] = np.sqrt((data['x'] - expected_src_pos_x_m)**2 + (data['y'] - expected_src_pos_y_m)**2)

    disp, miss = camera_to_shower_coordinates(
        expected_src_pos_x_m,
        expected_src_pos_y_m,
        data['x'],
        data['y'],
        data['psi']                                                                                                                                                                                                   )
    
    src_dep_params['time_gradient_from_source'] = data['time_gradient'] * np.sign(disp) * -1
    src_dep_params['skewness_from_source'] = data['skewness'] * np.sign(disp) * -1
    src_dep_params['alpha'] = np.rad2deg(np.arctan(np.abs(miss / disp)))

    return src_dep_params


def get_expected_source_pos(data, data_type, config, focal_length=28*u.m):

    """Get expected source position for source-dependent analysis .

    Parameters:
    -----------
    data: Pandas DataFrame
    data_type: string ('mc_gamma','mc_proton','real_data')
    config: dictionnary containing configuration
    
    """

    #For gamma MC, expected source position is actual one for each event
    if data_type == 'mc_gamma':
        expected_src_pos_x_m = data['src_x'].values
        expected_src_pos_y_m = data['src_y'].values

    #For proton MC, nominal source position is one written in config file
    if data_type == 'mc_proton':
        expected_src_pos = utils.sky_to_camera(
            u.Quantity(data['mc_alt_tel'].values + config['mc_nominal_source_x_deg'], u.deg, copy=False),
            u.Quantity(data['mc_az_tel'].values + config['mc_nominal_source_y_deg'], u.deg, copy=False),
            focal_length,
            u.Quantity(data['mc_alt_tel'].values, u.deg, copy=False),
            u.Quantity(data['mc_az_tel'].values, u.deg, copy=False)
        )
        
        expected_src_pos_x_m = expected_src_pos.x.to_value(u.m)
        expected_src_pos_y_m = expected_src_pos.y.to_value(u.m)

    # For real data
    if data_type == 'real_data':
        # source is always at the ceter of camera for ON mode
        if config.get('observation_mode') == 'on':
            expected_src_pos_x_m = np.zeros(len(data))
            expected_src_pos_y_m = np.zeros(len(data))
        
        # compute source position in camera coordinate event by event for wobble mode
        if config.get('observation_mode') == 'wobble':

            if 'source_name' in config:
                source_coord = SkyCoord.from_name(config.get('source_name'))
            else:
                source_coord = SkyCoord(config.get('source_ra'), config.get('source_dec'), frame="icrs", unit="deg")
            
            time = data['dragon_time']
            obstime = Time(time, scale='utc', format='unix')
            pointing_alt = u.Quantity(data['alt_tel'], u.rad, copy=False)
            pointing_az = u.Quantity(data['az_tel'],  u.rad, copy=False)
            source_pos = utils.radec_to_camera(source_coord, obstime, pointing_alt, pointing_az, focal_length)

            expected_src_pos_x_m = source_pos.x.to_value(u.m)
            expected_src_pos_y_m = source_pos.y.to_value(u.m)
   
    return expected_src_pos_x_m, expected_src_pos_y_m 
