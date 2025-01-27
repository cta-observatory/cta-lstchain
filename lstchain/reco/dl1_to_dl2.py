"""Module with functions for Energy and disp_norm reconstruction and G/H
separation. There are functions for raining random forest and for
applying them to data. The RF can be saved into a file for later use.

Usage:

"import dl1_to_dl2"
"""

import os
import logging

import astropy.units as u
import joblib
import numpy as np
import pandas as pd
import warnings
from astropy.coordinates import SkyCoord, Angle
from astropy.time import Time
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from . import disp
from . import utils
from ..io import (
    standard_config,
    replace_config,
    get_dataset_keys,
    get_srcdep_params,
)
from ..io.io import dl1_params_lstcam_key, dl1_params_src_dep_lstcam_key, dl1_likelihood_params_lstcam_key

from ctapipe.image.hillas import camera_to_shower_coordinates
from ctapipe.instrument import SubarrayDescription
from ctapipe.coordinates import CameraFrame, TelescopeFrame
from ctapipe_io_lst import OPTICS

logger = logging.getLogger(__name__)

__all__ = [
    'add_zd_interpolation_info',
    'apply_models',
    'build_models',
    'get_expected_source_pos',
    'get_source_dependent_parameters',
    'predict_with_zd_interpolation',
    'train_disp_norm',
    'train_disp_sign',
    'train_disp_vector',
    'train_energy',
    'train_reco',
    'train_sep',
    'update_disp_with_effective_focal_length'
]

def add_zd_interpolation_info(dl2table, training_pointings):
    """
    Compute necessary parameters for the interpolation of RF predictions
    between the zenith pointings of the MC data in the training sample on
    which the RFs were trained.

    Parameters
    ----------
    dl2table : pandas.DataFrame
        DataFrame containing DL2 information, including 'alt_tel' and 'az_tel'.
        Four columns will be added: alt0, alt1, w0, w1.
        alt0 and alt1 are the alt_tel values (telescope elevation, in radians) of
        the closest and second-closest training MC pointings (closest in elevation,
        on the same side of culmination) for each event in the table. The values
        w0 and w1 are the corresponding weights that, multiplied by the RF
        predictions at those two pointings, provide the interpolated result for
        each event's pointing.

    training_pointings : astropy.table.Table 
        Table containing the pointings (zd, az) of the MC training nodes.

    Returns
    -------
    pandas.DataFrame
        Updated DL2 pandas dataframe with additional columns alt0, alt1, w0, w1.

    """

    alt_tel = np.array(dl2table['alt_tel'])
    az_tel  = np.array(dl2table['az_tel'])

    training_alt_rad = np.pi / 2 - training_pointings['zd'].to(u.rad).value
    training_az_rad = training_pointings['az'].to(u.rad).value

    tiled_az = np.broadcast_to(az_tel[:, np.newaxis],
                               (len(dl2table), len(training_az_rad)))
    tiled_alt = np.broadcast_to(alt_tel[:, np.newaxis],
                                (len(dl2table), len(training_az_rad)))

    delta_alt = np.abs(training_alt_rad - tiled_alt)
    # mask to select training nodes only on the same side of the source
    # culmination as the event:
    same_side_of_culmination = np.sign(np.sin(training_az_rad) *
                                       np.sin(tiled_az)) > 0
    # Just fill a large value for pointings on the other side of culmination:
    delta_alt = np.where(same_side_of_culmination, delta_alt, np.pi/2)
    # indices ordered according to distance in telescope elevation
    sorted_indices = np.argsort(delta_alt, axis=1)
    closest_alt = training_alt_rad[sorted_indices[:, 0]]
    second_closest_alt = training_alt_rad[sorted_indices[:, 1]]

    c0 = np.cos(np.pi / 2 - closest_alt)
    c1 = np.cos(np.pi / 2 - second_closest_alt)
    cos_tel_zd = np.cos(np.pi / 2 - alt_tel)

    # Compute the weights w0, w1 that multiplied times the RF predictions at
    # the closest (0) and 2nd-closest (1) nodes (in alt_tel) result in the
    # interpolated value. Take care of cases in which the two closest nodes
    # happen to have the same zenith (or very close)! (if so, both nodes are
    # set to have equal weight in the interpolation)
    w1 = np.where(np.isclose(closest_alt, second_closest_alt, atol=1e-4, rtol=0),
                  0.5, (cos_tel_zd - c0) / (c1 - c0))
    w0 = 1 - w1

    # Update the dataframe:
    with pd.option_context('mode.copy_on_write', True):
        dl2table = dl2table.assign(alt0=closest_alt,
                                   alt1=second_closest_alt,
                                   w0=w0,
                                   w1=w1)

    return dl2table


def predict_with_zd_interpolation(rf, param_array, features):
    """
    Obtain a RF prediction which takes into account the difference between
    the telescope elevation (alt_tel, i.e. 90 deg - zenith) and those of the
    MC training nodes. The dependence of image parameters (for a shower of
    given characteristics) with zenith is strong at angles beyond ~50 deg,
    due to the change in airmass. Given the way Random Forests work, if the
    training is performed with a discrete distribution of pointings,
    the prediction of the RF will be biased for pointings in between those
    used in training. If zenith is used as one of the RF features, there will
    be a sudden jump in the predictions halfway between the training nodes.

    To solve this, we compute here two predictions for each event, one using
    the elevation (alt_tel) of the training pointing which is closest to the
    telescope pointing, and another one usimg the elevation of the
    sceond-closest pointing. Then the values are interpolated (linearly in
    cos(zenith)) to the actual zenith pointing (90 deg - alt_tel) of the event.

    Parameters
    ----------
    rf : sklearn.ensemble.RandomForestRegressor or RandomForestClassifier,
        The random forest we want to apply (must contain alt_tel among the
        training parameters).
    param_array : pandas.DataFrame
        Dataframe containing the features needed by the RF.
        It must also contain four additional columns: alt0, alt1, w0, w1, which
        can be added with the function add_zd_interpolation_info. These are the
        event-wise telescope elevations for the closest and 2nd-closest training
        pointings (alt0 and alt1), and the event-wise weights (w0 and w1) which
        must be applied to the RF prediction at the two pointings to obtain the
        interpolated value at the actual telescope pointing. Since the weights
        are the same (for a given event) for different RFs, it does not make
        sense to compute them here - they are pre-calculated by
        `add_zd_interpolation_info`.
    features : list of str
        List of the names of the image features used by the RF.

    Return
    ------
    numpy.ndarray
        Interpolated RF predictions. 1D array for regressors (log energy,
        or disp_norm), 2D (events, # of classes) for classifiers.

    """

    # Type of RF (classifier or regressor):
    is_classifier = isinstance(rf, RandomForestClassifier)

    features_copy = features.copy()
    alt_index_in_features = features_copy.index('alt_tel')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # This is just to avoid the RFs to warn about the features
        # unnamed (passed as an array). We do this because we want to replace
        # alt_tel by alt0, then by alt1...
        # First use alt_tel of closest MC training node:
        features_copy[alt_index_in_features] = 'alt0'
        if is_classifier:
            prediction_0 = rf.predict_proba(param_array[features_copy].to_numpy())
        else:
            prediction_0 = rf.predict(param_array[features_copy].to_numpy())
        # Now the alt_tel value of the second-closest node:
        features_copy[alt_index_in_features] = 'alt1'
        if is_classifier:
            prediction_1 = rf.predict_proba(param_array[features_copy].to_numpy())
        else:
            prediction_1 = rf.predict(param_array[features_copy].to_numpy())

    # Interpolated RF prediction:
    if is_classifier:
        prediction = (prediction_0.T * param_array['w0'].values +
                      prediction_1.T * param_array['w1'].values).T
    else:
        prediction = (prediction_0 * param_array['w0'] +
                      prediction_1 * param_array['w1']).values

    return prediction

def train_energy(train, custom_config=None):
    """
    Train a Random Forest Regressor for the regression of the energy
    TODO: introduce the possibility to use another model

    Parameters
    ----------
    train: `pandas.DataFrame`
    custom_config : dict
        Modified configuration to update the standard one

    Returns
    -------
    The trained model
    """
    custom_config = {} if custom_config is None else custom_config
    config = replace_config(standard_config, custom_config)
    energy_regression_args = config['random_forest_energy_regressor_args']
    features = config['energy_regression_features']
    model = RandomForestRegressor

    logger.info("Given features: ", features)
    logger.info("Number of events for training: ", train.shape[0])
    logger.info("Training Random Forest Regressor for Energy Reconstruction...")

    reg = model(**energy_regression_args)
    reg.fit(train[features],
            train['log_mc_energy'],
            sample_weight=train['weight'])

    logger.info("Model {} trained!".format(model))
    return reg


def train_disp_vector(train, custom_config=None, predict_features=None):
    """
    Train a model (Random Forest Regressor) for the regression of the disp_norm vector coordinates dx,dy.
    Therefore, the model must be able to be applied on a vector of features.
    TODO: introduce the possibility to use another model

    Parameters
    ----------
    train: `pandas.DataFrame`
    custom_config: dictionnary
        Modified configuration to update the standard one. Default=None
    predict_features: list
        list of predict features names. Default=['disp_dx', 'disp_dy']

    Returns
    -------
    The trained model
    """
    if predict_features is None:
        predict_features = ['disp_dx', 'disp_dy']
    custom_config = {} if custom_config is None else custom_config
    config = replace_config(standard_config, custom_config)
    disp_regression_args = config['random_forest_disp_regressor_args']
    features = config['disp_regression_features']
    model = RandomForestRegressor

    logger.info("Given features: ", features)
    logger.info("Number of events for training: ", train.shape[0])
    logger.info("Training model {} for disp vector regression".format(model))

    reg = model(**disp_regression_args)
    x = train[features]
    y = np.transpose([train[f] for f in predict_features])
    reg.fit(x, y, sample_weight=train['weight'])

    logger.info("Model {} trained!".format(model))

    return reg


def train_disp_norm(train, custom_config=None, predict_feature='disp_norm'):
    """
    Train a model for the regression of the disp_norm norm

    Parameters
    ----------
    train: `pandas.DataFrame`
    custom_config: dictionnary
        Modified configuration to update the standard one

    Returns
    -------
    The trained model
    """
    custom_config = {} if custom_config is None else custom_config
    config = replace_config(standard_config, custom_config)
    disp_regression_args = config['random_forest_disp_regressor_args']
    features = config['disp_regression_features']
    model = RandomForestRegressor

    logger.info("Given features: ", features)
    logger.info("Number of events for training: ", train.shape[0])
    logger.info("Training model {} for disp norm regression".format(model))

    reg = model(**disp_regression_args)
    x = train[features]
    y = np.transpose(train[predict_feature])
    reg.fit(x, y, sample_weight=train['weight'])

    logger.info("Model {} trained!".format(model))

    return reg


def train_disp_sign(train, custom_config=None, predict_feature='disp_sign'):
    """
    Train a model for the classification of the disp_norm sign

    Parameters
    ----------
    train: `pandas.DataFrame`
    custom_config: dictionnary
        Modified configuration to update the standard one

    Returns
    -------
    The trained model
    """
    custom_config = {} if custom_config is None else custom_config
    config = replace_config(standard_config, custom_config)
    classification_args = config['random_forest_disp_classifier_args']
    features = config["disp_classification_features"]
    model = RandomForestClassifier

    logger.info("Given features: ", features)
    logger.info("Number of events for training: ", train.shape[0])
    logger.info("Training model {} for disp sign classification".format(model))

    clf = model(**classification_args)
    x = train[features]
    y = np.transpose(train[predict_feature])
    clf.fit(x, y, sample_weight=train['weight'])

    logger.info("Model {} trained!".format(model))

    return clf


def train_reco(train, custom_config=None):
    """
    Trains two Random Forest regressors for Energy and disp_norm
    reconstruction respectively. Returns the trained RF.

    Parameters
    ----------
    train: `pandas.DataFrame`
    custom_config: dictionnary
        Modified configuration to update the standard one

    Returns
    -------
    RandomForestRegressor: reg_energy
    RandomForestRegressor: reg_disp
    """
    custom_config = {} if custom_config is None else custom_config
    config = replace_config(standard_config, custom_config)
    energy_regression_args = config['random_forest_energy_regressor_args']
    disp_regression_args = config['random_forest_disp_regressor_args']
    energy_features = config['energy_regression_features']
    disp_features = config['disp_regression_features']
    model = RandomForestRegressor

    logger.info("Given energy_features: ", energy_features)
    logger.info("Number of events for training: ", train.shape[0])
    logger.info("Training Random Forest Regressor for Energy Reconstruction...")

    reg_energy = model(**energy_regression_args)
    reg_energy.fit(train[energy_features],
                   train['log_mc_energy'], sample_weight = train['weight']
    )

    logger.info("Random Forest trained!")
    logger.info("Given disp_features: ", disp_features)
    logger.info("Training Random Forest Regressor for disp_norm Reconstruction...")

    reg_disp = RandomForestRegressor(**disp_regression_args)
    reg_disp.fit(train[disp_features],
                 train['disp_norm'],
                 sample_weight=train['weight'])

    logger.info("Random Forest trained!")
    logger.info("Done!")
    return reg_energy, reg_disp


def train_sep(train, custom_config=None):
    """Trains a Random Forest classifier for Gamma/Hadron separation.
    Returns the trained RF.

    Parameters
    ----------
    train: `pandas.DataFrame`
        data set for training the RF
    custom_config: dict
        Modified configuration to update the standard one

    Returns
    -------
    `RandomForestClassifier`
    """
    custom_config = {} if custom_config is None else custom_config
    config = replace_config(standard_config, custom_config)
    classification_args = config['random_forest_particle_classifier_args']
    features = config["particle_classification_features"]
    model = RandomForestClassifier

    logger.info("Given features: ", features)
    logger.info("Number of events for training: ", train.shape[0])
    logger.info("Training Random Forest Classifier for",
          "Gamma/Hadron separation...")

    clf = model(**classification_args)

    clf.fit(train[features],
            train['mc_type'],
            sample_weight=train['weight'])
    logger.info("Random Forest trained!")
    return clf


def build_models(filegammas, fileprotons,
                 save_models=True, path_models="./",
                 free_model_memory=True,
                 energy_min=-np.inf,
                 custom_config=None,
                 ):
    """
    Uses MC data to train Random Forests for Energy and DISP
    reconstruction and G/H separation and returns the trained RFs.
    The passed config superseeds the standard configuration.
    Here is the complete workflow with the number of events selected from the config:

    .. mermaid::

        graph LR
            GAMMA[gammas] -->|#`gamma_regressors`| REG(regressors) --> DISK
            GAMMA --> S(split)
            S --> |#`gamma_tmp_regressors`| g_train
            S --> |#`gamma_classifier`| g_test
            g_train --> tmp_reg(tmp regressors)
            tmp_reg --- A[ ]:::empty
            g_test --- A
            A --> g_test_dl2
            g_test_dl2 --- D[ ]:::empty
            protons -------- |#`proton_classifier`| D
            D --> cls(classifier)
            cls--> DISK
            classDef empty width:0px,height:0px;


    Parameters
    ----------
    filegammas: string
        path to the file with MC gamma events
    fileprotons: string
        path to the file with MC proton events
    save_models: bool
        True to save the trained models on disk
    path_models: string
        path of a directory where to save the models.
        if it does exist, the directory is created
    free_model_memory: bool
        If True RF models are freed after use and not returned
    energy_min: float
        Cut in intensity of the showers for training RF
    custom_config: dictionnary
       Modified configuration to update the standard one
    test_size: float or int
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
        If int, represents the absolute number of test samples.
        If None, it will be set to 0.25.

    Returns
    -------
    if config['disp_method'] == 'disp_vector':
        return reg_energy, reg_disp_vector, cls_gh
    elif config['disp_method'] == 'disp_norm_sign':
        return reg_energy, reg_disp_norm, cls_disp_sign, cls_gh

    Raises
    ------
    ValueError
        If the requested number of gamma events in the config for the training of the classifier is not valid.
        See config["n_training_events"]
    """

    custom_config = {} if custom_config is None else custom_config
    config = replace_config(standard_config, custom_config)
    events_filters = config["events_filters"]

    # Adding a filter on mc_type just for training
    events_filters['mc_type'] = [-9000, np.inf]

    pointing_wise_weights = False
    if 'random_forest_weight_settings' in config:
        if config['random_forest_weight_settings']['pointing_wise_weights']:
            logger.info("Pointing-wise event weighting activated")
            pointing_wise_weights = True

    df_gamma = pd.read_hdf(filegammas, key=dl1_params_lstcam_key)
    df_proton = pd.read_hdf(fileprotons, key=dl1_params_lstcam_key)

    # Update parameters related to target direction on camera frame for gamma MC
    # taking into account of the aberration effect using effective focal length
    try:
        subarray_info = SubarrayDescription.from_hdf(filegammas)
        tel_id = config["allowed_tels"][0] if "allowed_tels" in config else 1
        effective_focal_length = subarray_info.tel[tel_id].optics.effective_focal_length
    except OSError:
        logger.warning("subarray table is not readable because of the version incompatibility.")
        logger.warning("The effective focal length for the standard LST optics will be used.")
        effective_focal_length = OPTICS.effective_focal_length

    df_gamma = update_disp_with_effective_focal_length(df_gamma, effective_focal_length=effective_focal_length)
    if 'lh_fit_config' in config.keys():
        lhfit_df_gamma = pd.read_hdf(filegammas, key=dl1_likelihood_params_lstcam_key)
        df_gamma = pd.concat([df_gamma, lhfit_df_gamma], axis=1)
        lhfit_df_proton = pd.read_hdf(fileprotons, key=dl1_likelihood_params_lstcam_key)
        df_proton = pd.concat([df_proton, lhfit_df_proton], axis=1)

    if config['source_dependent']:
        # if source-dependent parameters are already in dl1 data, just read those data
        # if not, source-dependent parameters are added here
        if dl1_params_src_dep_lstcam_key in get_dataset_keys(filegammas):
            src_dep_df_gamma = get_srcdep_params(filegammas)

        else:
            src_dep_df_gamma = get_source_dependent_parameters(
                df_gamma, config, effective_focal_length=effective_focal_length
            )

        df_gamma = pd.concat([df_gamma, src_dep_df_gamma['on']], axis=1)

        # if source-dependent parameters are already in dl1 data, just read those data
        # if not, source-dependent parameters are added here
        if dl1_params_src_dep_lstcam_key in get_dataset_keys(fileprotons):
            src_dep_df_proton = get_srcdep_params(fileprotons)

        else:
            try:
                subarray_info = SubarrayDescription.from_hdf(fileprotons)
                tel_id = config["allowed_tels"][0] if "allowed_tels" in config else 1
                effective_focal_length = subarray_info.tel[tel_id].optics.effective_focal_length
            except OSError:
                logger.warning("subarray table is not readable because of the version incompatibility.")
                logger.warning("The effective focal length for the standard LST optics will be used.")
                effective_focal_length = OPTICS.effective_focal_length

            src_dep_df_proton = get_source_dependent_parameters(
                df_proton, config, effective_focal_length=effective_focal_length
            )

        df_proton = pd.concat([df_proton, src_dep_df_proton['on']], axis=1)

    # Normalize all azimuth angles to the range [0, 360) degrees
    df_gamma.az_tel = Angle(df_gamma.az_tel, u.rad).wrap_at(360 * u.deg).rad
    df_proton.az_tel = Angle(df_proton.az_tel, u.rad).wrap_at(360 * u.deg).rad

    # Dealing with `sin_az_tel` missing data because of the former version of lstchain
    if 'sin_az_tel' not in df_gamma.columns:
        df_gamma['sin_az_tel'] = np.sin(df_gamma.az_tel)
    if 'sin_az_tel' not in df_proton.columns:
        df_proton['sin_az_tel'] = np.sin(df_proton.az_tel)

    df_gamma = utils.filter_events(df_gamma,
                                   filters=events_filters,
                                   finite_params=config['energy_regression_features']
                                                 + config['disp_regression_features']
                                                 + config['particle_classification_features']
                                                 + config['disp_classification_features'],
                                   )

    df_proton = utils.filter_events(df_proton,
                                    filters=events_filters,
                                    finite_params=config['energy_regression_features']
                                                  + config['disp_regression_features']
                                                  + config['particle_classification_features']
                                                  + config['disp_classification_features'],
                                    )

    # Training MC gammas in reduced viewcone
    src_r_min = config['train_gamma_src_r_deg'][0]
    src_r_max = config['train_gamma_src_r_deg'][1]
    df_gamma = utils.apply_src_r_cut(df_gamma, src_r_min, src_r_max)

    if pointing_wise_weights:
        # Give same total weight to all events in every pointing node, by
        # applying event-wise weights which depend on the statistics per node
        # The weight is written to a new column of df_gamma, called 'weight'
        _, _ = utils.compute_rf_event_weights(df_gamma)
    else:
        df_gamma['weight'] = np.ones(len(df_gamma))

    # Train regressors for energy and disp_norm reconstruction, only with gammas
    n_gamma_regressors = config["n_training_events"]["gamma_regressors"]
    if n_gamma_regressors not in [1.0, None]:
        try:
            df_gamma_reg, _ = train_test_split(df_gamma, train_size=n_gamma_regressors)
        except ValueError as e:
            raise ValueError(f"The requested number of gammas {n_gamma_regressors} "
                             f"for the regressors training is not valid.") from e
    else:
        df_gamma_reg = df_gamma

    reg_energy = train_energy(df_gamma_reg, custom_config=config)

    if save_models:
        os.makedirs(path_models, exist_ok=True)

        file_reg_energy = path_models + "/reg_energy.sav"
        joblib.dump(reg_energy, file_reg_energy, compress=3)
    if free_model_memory:
        del reg_energy


    if config['disp_method'] == 'disp_vector':
        reg_disp_vector = train_disp_vector(df_gamma, custom_config=config)
        if save_models:
            file_reg_disp_vector = path_models + "/reg_disp_vector.sav"
            joblib.dump(reg_disp_vector, file_reg_disp_vector, compress=3)
        if free_model_memory:
            del reg_disp_vector
    elif config['disp_method'] == 'disp_norm_sign':
        reg_disp_norm = train_disp_norm(df_gamma, custom_config=config)
        cls_disp_sign = train_disp_sign(df_gamma, custom_config=config)
        if save_models:
            file_reg_disp_norm = os.path.join(path_models, 'reg_disp_norm.sav')
            file_cls_disp_sign = os.path.join(path_models, 'cls_disp_sign.sav')
            joblib.dump(reg_disp_norm, file_reg_disp_norm, compress=3)
            joblib.dump(cls_disp_sign, file_cls_disp_sign, compress=3)
        if free_model_memory:
            del reg_disp_norm
            del cls_disp_sign

    # Train classifier for gamma/hadron separation.
    test_size = config['n_training_events']['gamma_classifier']
    train_size = config['n_training_events']['gamma_tmp_regressors']
    try:
        train, testg = train_test_split(df_gamma, test_size=test_size, train_size=train_size)
    except ValueError as e:
        raise ValueError(
            "The requested number of gammas for the classifier training is not valid."
        ) from e

    n_proton_classifier = config["n_training_events"]["proton_classifier"]
    if n_proton_classifier not in [1.0, None]:
        try:
            df_proton, _ = train_test_split(df_proton, train_size=config['n_training_events']['proton_classifier'])
        except ValueError as e:
            raise ValueError(
                "The requested number of protons for the classifier training is not valid."
            ) from e

    if pointing_wise_weights:
        # Give same total weight to all events in every pointing node, by
        # applying event-wise weights which depend on the statistics per node
        # The weight is written to a new column of df_gamma, called 'weight'
        _, _ = utils.compute_rf_event_weights(df_proton)
    else:
        df_proton['weight'] = np.ones(len(df_proton))

    test = pd.concat([testg, df_proton], ignore_index=True)

    temp_reg_energy = train_energy(train, custom_config=config)
    # Apply the temporary energy regressors to the test set
    test['log_reco_energy'] = temp_reg_energy.predict(test[config['energy_regression_features']])
    del temp_reg_energy

    if config['disp_method'] == 'disp_vector':
        temp_reg_disp_vector = train_disp_vector(train, custom_config=config)
        # Apply the temporary disp vector regressors to the test set
        disp_vector = temp_reg_disp_vector.predict(test[config['disp_regression_features']])
        del temp_reg_disp_vector
    elif config['disp_method'] == 'disp_norm_sign':
        tmp_reg_disp_norm = train_disp_norm(train, custom_config=config)
        tmp_cls_disp_sign = train_disp_sign(train, custom_config=config)
        # Apply the temporary disp norm regressor and sign classifier to the test set
        disp_norm = tmp_reg_disp_norm.predict(test[config['disp_regression_features']])
        disp_sign_proba = tmp_cls_disp_sign.predict_proba(test[config['disp_classification_features']])
        col = list(tmp_cls_disp_sign.classes_).index(1)
        disp_sign = np.where(disp_sign_proba[:, col] > 0.5, 1, -1)

        del tmp_reg_disp_norm
        del tmp_cls_disp_sign
        test['reco_disp_norm'] = disp_norm
        test['reco_disp_sign'] = disp_sign
        test['reco_disp_sign_proba'] = disp_sign_proba[:, 0]
        
        disp_angle = test['psi']  # the source here is supposed to be in the direction given by Hillas
        disp_vector = disp.disp_vector(disp_norm, disp_angle, disp_sign)

    test['reco_disp_dx'] = disp_vector[:, 0]
    test['reco_disp_dy'] = disp_vector[:, 1]

    test['reco_src_x'], test['reco_src_y'] = disp.disp_to_pos(test['reco_disp_dx'],
                                                              test['reco_disp_dy'],
                                                              test['x'], test['y'])

    # give skewness and time gradient a meaningful sign, i.e. referred to the reconstructed source position:
    longi, _ = camera_to_shower_coordinates(test['reco_src_x'], test['reco_src_y'],
                                            test['x'], test['y'], test['psi'])
    test['signed_skewness'] = -1 * np.sign(longi) * test['skewness']
    test['signed_time_gradient'] = -1 * np.sign(longi) * test['time_gradient']

    # Apply cut in reconstructed energy. New train set is the previous
    # test with energy and disp_norm reconstructed.

    train = test[test['log_reco_energy'] > energy_min]

    # source-dep & indep combined parameters
    if config['source_dependent']:
        train['reco_disp_sign_correctness'] = train['reco_disp_sign_proba']
        select = np.sign(train['skewness']) * np.sign(train['skewness_from_source']) == -1
        train['reco_disp_sign_correctness'][select] = 1 - train['reco_disp_sign_correctness'][select]

        train['reco_disp_norm_diff'] = np.abs(train['dist'] - train['reco_disp_norm'])

    # Check that any new features used in particle classification are finite
    train = utils.filter_events(train,
                                filters=events_filters,
                                finite_params=config['particle_classification_features']
                                )
    # Train the Classifier
    cls_gh = train_sep(train, custom_config=config)

    if save_models:
        file_cls_gh = path_models + "/cls_gh.sav"
        joblib.dump(cls_gh, file_cls_gh, compress=3)
    if free_model_memory:
        del cls_gh

    if not free_model_memory:
        if config['disp_method'] == 'disp_vector':
            return reg_energy, reg_disp_vector, cls_gh
        elif config['disp_method'] == 'disp_norm_sign':
            return reg_energy, reg_disp_norm, cls_disp_sign, cls_gh


def apply_models(dl1,
                 classifier,
                 reg_energy,
                 reg_disp_vector=None,
                 reg_disp_norm=None,
                 cls_disp_sign=None,
                 effective_focal_length=29.30565 * u.m,
                 custom_config=None,
                 interpolate_rf=None,
                 training_pointings=None
                 ):
    """
    Apply previously trained Random Forests to a set of data
    depending on a set of features.
    The right set of disp models must be passed depending on the config.

    Parameters
    ----------
    dl1: `pandas.DataFrame`
    classifier: string | Path | bytes | sklearn.ensemble.RandomForestClassifier
        Path to the random forest filename or file or pre-loaded RandomForestClassifier object
        for Gamma/Hadron separation
    reg_energy: string | Path | bytes | sklearn.ensemble.RandomForestRegressor
        Path to the random forest filename or file or pre-loaded RandomForestRegressor object
        for Energy reconstruction
    reg_disp_vector: string | Path | bytes | sklearn.ensemble.RandomForestRegressor
        Path to the random forest filename or file or pre-loaded RandomForestRegressor object
        for disp vector reconstruction
    reg_disp_norm: string | Path | bytes | sklearn.ensemble.RandomForestRegressor
        Path to the random forest filename or file or pre-loaded RandomForestRegressor object
        for disp norm reconstruction
    cls_disp_sign: string | Path | bytes | sklearn.ensemble.RandomForestClassifier
        Path to the random forest filename or file or pre-loaded RandomForestClassifier object
        for disp sign reconstruction
    effective_focal_length: `astropy.unit`
    custom_config: dictionary
        Modified configuration to update the standard one
    interpolate_rf : dict
        Contains three booleans, 'energy_regression',
        'particle_classification', 'disp', indicating which RF predictions
        should be interpolated linearly in cos(zenith).
    training_pointings : astropy.table.Table 
        Table with azimuth (az), zenith (zd) pointings of the MC sample used
        in the training. Needed for the interpolation of RF predictions.

    Returns
    -------
    `pandas.DataFrame`
        dataframe including reconstructed dl2 features
    """
    custom_config = {} if custom_config is None else custom_config
    config = replace_config(standard_config, custom_config)
    energy_regression_features = config["energy_regression_features"]
    disp_regression_features = config["disp_regression_features"]
    disp_classification_features = config["disp_classification_features"]
    classification_features = config["particle_classification_features"]
    events_filters = config["events_filters"]

    # If no settings are provided for RF interpolation, it is switched off:
    if interpolate_rf is None:
        interpolate_rf = {'energy_regression': False,
                          'particle_classification': False,
                          'disp': False}

    dl2 = utils.filter_events(dl1,
                              filters=events_filters,
                              finite_params=config['disp_regression_features']
                                            + config['energy_regression_features']
                                            + config['particle_classification_features']
                                            + config['disp_classification_features'],
                              )
    # if no events in dl2, e.g. for bad time interval from Cat-B calibration
    if len(dl2) == 0:
        logger.warning("No events in dl2.")
        return dl2
    
    # Update parameters related to target direction on camera frame for MC data
    # taking into account of the abrration effect using effective focal length
    is_simu = 'disp_norm' in dl2.columns
    if is_simu:
        dl2 = update_disp_with_effective_focal_length(dl2,
                                                      effective_focal_length=effective_focal_length)

    if True in interpolate_rf.values():
        # Interpolation of RF predictions is switched on
        dl2 = add_zd_interpolation_info(dl2, training_pointings)

    # Reconstruction of Energy and disp_norm distance
    if isinstance(reg_energy, (str, bytes, Path)):
        reg_energy = joblib.load(reg_energy)
    if interpolate_rf['energy_regression']:
        # Interpolation of RF predictions (linear in cos(zenith)):
        dl2['log_reco_energy'] = predict_with_zd_interpolation(reg_energy, dl2,
                                                               energy_regression_features)
    else:
        dl2['log_reco_energy'] = reg_energy.predict(dl2[energy_regression_features])
    del reg_energy
    dl2['reco_energy'] = 10 ** (dl2['log_reco_energy'])

    if config['disp_method'] == 'disp_vector':
        if isinstance(reg_disp_vector, (str, bytes, Path)):
            reg_disp_vector = joblib.load(reg_disp_vector)
        if interpolate_rf['disp']:
            disp_vector = predict_with_zd_interpolation(reg_disp_vector, dl2,
                                                        disp_regression_features)
        else:
            disp_vector = reg_disp_vector.predict(dl2[disp_regression_features])
        del reg_disp_vector
    elif config['disp_method'] == 'disp_norm_sign':
        if isinstance(reg_disp_norm, (str, bytes, Path)):
            reg_disp_norm = joblib.load(reg_disp_norm)
        if interpolate_rf['disp']:
            disp_norm = predict_with_zd_interpolation(reg_disp_norm, dl2,
                                                      disp_regression_features)
        else:
            disp_norm = reg_disp_norm.predict(dl2[disp_regression_features])
        del reg_disp_norm

        if isinstance(cls_disp_sign, (str, bytes, Path)):
            cls_disp_sign = joblib.load(cls_disp_sign)
        if interpolate_rf['disp']:
            disp_sign_proba = predict_with_zd_interpolation(cls_disp_sign, dl2,
                                                            disp_classification_features)
        else:
            disp_sign_proba = cls_disp_sign.predict_proba(dl2[disp_classification_features])

        col = list(cls_disp_sign.classes_).index(1)
        disp_sign = np.where(disp_sign_proba[:, col] > 0.5, 1, -1)
        del cls_disp_sign

        dl2['reco_disp_norm'] = disp_norm
        dl2['reco_disp_sign'] = disp_sign
        dl2['reco_disp_sign_proba'] = disp_sign_proba[:, 0]
        
        disp_angle = dl2['psi']  # the source here is supposed to be in the direction given by Hillas
        disp_vector = disp.disp_vector(disp_norm, disp_angle, disp_sign)

    dl2['reco_disp_dx'] = disp_vector[:, 0]
    dl2['reco_disp_dy'] = disp_vector[:, 1]

    # Construction of Source position in camera coordinates from disp_norm distance.

    dl2['reco_src_x'], dl2['reco_src_y'] = disp.disp_to_pos(dl2.reco_disp_dx,
                                                            dl2.reco_disp_dy,
                                                            dl2.x,
                                                            dl2.y,
                                                            )

    longi, _ = camera_to_shower_coordinates(dl2['reco_src_x'], dl2['reco_src_y'],
                                            dl2['x'], dl2['y'], dl2['psi'])

    # Obtain the time gradient with sign relative to the reconstructed shower direction (reco_src_x, reco_src_y)
    # Defined positive if light arrival times increase with distance to it. Negative otherwise:
    dl2['signed_time_gradient'] = -1 * np.sign(longi) * dl2['time_gradient']

    # Obtain skewness with sign relative to the reconstructed shower direction (reco_src_x, reco_src_y)
    # Defined on the major image axis; sign is such that it is typically positive for gammas:
    dl2['signed_skewness'] = -1 * np.sign(longi) * dl2['skewness']

    if 'mc_alt_tel' in dl2.columns:
        alt_tel = dl2['mc_alt_tel'].values
        az_tel = dl2['mc_az_tel'].values
    elif 'alt_tel' in dl2.columns:
        alt_tel = dl2['alt_tel'].values
        az_tel = dl2['az_tel'].values
    else:
        alt_tel = - np.pi / 2. * np.ones(len(dl2))
        az_tel = - np.pi / 2. * np.ones(len(dl2))

    src_pos_reco = utils.reco_source_position_sky(dl2.x.values * u.m,
                                                  dl2.y.values * u.m,
                                                  dl2.reco_disp_dx.values * u.m,
                                                  dl2.reco_disp_dy.values * u.m,
                                                  effective_focal_length,
                                                  alt_tel * u.rad,
                                                  az_tel * u.rad)

    dl2['reco_alt'] = src_pos_reco.alt.rad
    dl2['reco_az'] = src_pos_reco.az.rad

    # source-dep & indep combined parameters    
    if config['source_dependent']:
        dl2['reco_disp_sign_correctness'] = dl2['reco_disp_sign_proba']
        select = np.sign(dl2['skewness']) * np.sign(dl2['skewness_from_source']) == -1
        dl2['reco_disp_sign_correctness'][select] = 1 - dl2['reco_disp_sign_correctness'][select]

        dl2['reco_disp_norm_diff'] = np.abs(dl2['dist'] - dl2['reco_disp_norm'])
        
    
    if isinstance(classifier, (str, bytes, Path)):
        classifier = joblib.load(classifier)
    if interpolate_rf['particle_classification']:
        probs = predict_with_zd_interpolation(classifier, dl2,
                                              classification_features)
    else:
        probs = classifier.predict_proba(dl2[classification_features])

    # This check is valid as long as we train on only two classes (gammas and protons)
    if probs.shape[1] > 2:
        raise ValueError("The classifier is predicting more than two classes, "
                         "the predicted probabilty to assign as gammaness is unclear."
                         "Please check training data")

    # gammaness is the prediction probability for the class 0 (proton: class 101)
    mc_type_gamma, mc_type_proton = 0, 101
    col = list(classifier.classes_).index(mc_type_gamma)
    dl2['gammaness'] = probs[:, col]
    dl2['reco_type'] = np.where(probs[:, col] > 0.5, mc_type_gamma, mc_type_proton)
    del classifier

    return dl2


def get_source_dependent_parameters(data, config, effective_focal_length=29.30565 * u.m):
    """Get parameters dict for source-dependent analysis.

    Parameters
    ----------
    data: Pandas DataFrame
    config: dictionnary containing configuration
    """

    is_simu = (data['mc_type'] >= 0).all() if 'mc_type' in data.columns else False

    if is_simu:
        data_type = 'mc_gamma' if (data['mc_type'] == 0).all() else 'mc_proton'
    else:
        data_type = 'real_data'

    expected_src_pos_x_m, expected_src_pos_y_m = get_expected_source_pos(
        data, data_type, config, effective_focal_length=effective_focal_length
    )

    src_dep_params = calc_source_dependent_parameters(data, expected_src_pos_x_m, expected_src_pos_y_m)
    src_dep_params_dict = {'on': src_dep_params}
    if not is_simu and config.get('observation_mode') == 'wobble':
        for ioff in range(config.get('n_off_wobble')):
            off_angle = 2 * np.pi / (config['n_off_wobble'] + 1) * (ioff + 1)

            rotated_expected_src_pos_x_m = expected_src_pos_x_m * np.cos(off_angle) - expected_src_pos_y_m * np.sin(
                off_angle)
            rotated_expected_src_pos_y_m = expected_src_pos_x_m * np.sin(off_angle) + expected_src_pos_y_m * np.cos(
                off_angle)
            src_dep_params = calc_source_dependent_parameters(data, rotated_expected_src_pos_x_m,
                                                              rotated_expected_src_pos_y_m)
            src_dep_params['off_angle'] = np.rad2deg(off_angle)
            src_dep_params_dict['off_{:03}'.format(round(np.rad2deg(off_angle)))] = src_dep_params

    return src_dep_params_dict


def calc_source_dependent_parameters(data, expected_src_pos_x_m, expected_src_pos_y_m):
    """Calculate source-dependent parameters with a given source position.

    Parameters
    ----------
    data: Pandas DataFrame
    expected_src_pos_x_m: float
    expected_src_pos_y_m: float
    """
    src_dep_params = pd.DataFrame(index=data.index)

    src_dep_params['expected_src_x'] = expected_src_pos_x_m
    src_dep_params['expected_src_y'] = expected_src_pos_y_m

    disp, miss = camera_to_shower_coordinates(
        expected_src_pos_x_m,
        expected_src_pos_y_m,
        data['x'],
        data['y'],
        data['psi'])

    src_dep_params['time_gradient_from_source'] = data['time_gradient'] * np.sign(disp) * -1
    src_dep_params['skewness_from_source'] = data['skewness'] * np.sign(disp) * -1

    if 'lhfit_x' in data.keys():
        # Use lhfit parameters for 'dist' and 'alpha'
        disp, miss = camera_to_shower_coordinates(
            expected_src_pos_x_m,
            expected_src_pos_y_m,
            data['lhfit_x'],
            data['lhfit_y'],
            data['lhfit_psi'])
        src_dep_params['dist'] = np.sqrt((data['lhfit_x'] - expected_src_pos_x_m) ** 2 +
                                         (data['lhfit_y'] - expected_src_pos_y_m) ** 2)
    else:
        src_dep_params['dist'] = np.sqrt((data['x'] - expected_src_pos_x_m) ** 2 +
                                         (data['y'] - expected_src_pos_y_m) ** 2)

    src_dep_params['alpha'] = np.rad2deg(np.arctan(np.abs(miss / disp)))

    return src_dep_params


def get_expected_source_pos(data, data_type, config, effective_focal_length=29.30565 * u.m):
    """Get expected source position for source-dependent analysis .

    Parameters
    ----------
    data: Pandas DataFrame
    data_type: string ('mc_gamma','mc_proton','real_data')
    config: dictionnary containing configuration
    """

    # For gamma MC, expected source position is actual one for each event
    if data_type == 'mc_gamma':
        data = update_disp_with_effective_focal_length(data, effective_focal_length = effective_focal_length)
        expected_src_pos_x_m = data['src_x'].values
        expected_src_pos_y_m = data['src_y'].values

    # For proton MC, nominal source position is one written in config file
    if data_type == 'mc_proton':
        source_pos = SkyCoord(
            fov_lon = -1 * config['mc_nominal_source_y_deg'] * u.deg,
            fov_lat = config['mc_nominal_source_x_deg'] * u.deg,
            frame=TelescopeFrame()
        )
        camera_frame = CameraFrame(focal_length=effective_focal_length)
        source_camera = source_pos.transform_to(camera_frame)
        expected_src_pos_x_m = source_camera.x.to_value(u.m)
        expected_src_pos_y_m = source_camera.y.to_value(u.m)

    # For real data
    if data_type == 'real_data':
        # source is always at the ceter of camera for ON mode
        if config.get('observation_mode') == 'on':
            expected_src_pos_x_m = np.zeros(len(data))
            expected_src_pos_y_m = np.zeros(len(data))

        # compute source position in camera coordinate event by event for wobble mode
        elif config.get('observation_mode') == 'wobble':

            if 'source_name' in config:
                source_coord = SkyCoord.from_name(config.get('source_name'))
            elif 'source_ra' and 'source_dec' in config:
                source_coord = SkyCoord(config.get('source_ra'), config.get('source_dec'), frame="icrs", unit="deg")
            else:
                raise KeyError(
                    'source position (`source_name` or `source_ra` & `source_dec`) is not defined in a config file for source-dependent analysis.'
                )

            time = data['dragon_time']
            obstime = Time(time, scale='utc', format='unix')
            pointing_alt = u.Quantity(data['alt_tel'], u.rad, copy=False)
            pointing_az = u.Quantity(data['az_tel'], u.rad, copy=False)
            source_pos = utils.radec_to_camera(source_coord, obstime, pointing_alt, pointing_az, effective_focal_length)

            expected_src_pos_x_m = source_pos.x.to_value(u.m)
            expected_src_pos_y_m = source_pos.y.to_value(u.m)
            
        else:
            raise KeyError(
                '`observation_mode` is not defined in a config file for source-dependent analysis. It should be `on` or `wobble`'
            )

    return expected_src_pos_x_m, expected_src_pos_y_m


def update_disp_with_effective_focal_length(data, effective_focal_length=29.30565 * u.m):
    """Update disp parameters using effective focal length

    Parameters
    ----------
    data: Pandas DataFrame
    config: dictionnary containing configuration
    """

    source_pos_in_camera = utils.sky_to_camera(
        u.Quantity(data['mc_alt'].values, u.rad, copy=False),
        u.Quantity(data['mc_az'].values, u.rad, copy=False),
        effective_focal_length,
        u.Quantity(data['mc_alt_tel'].values, u.rad, copy=False),
        u.Quantity(data['mc_az_tel'].values, u.rad, copy=False)
    )
    
    expected_src_pos_x_m = source_pos_in_camera.x.to_value(u.m)
    expected_src_pos_y_m = source_pos_in_camera.y.to_value(u.m)
    
    data['src_x'] = expected_src_pos_x_m
    data['src_y'] = expected_src_pos_y_m

    disp_dx, disp_dy, disp_norm, disp_angle, disp_sign = disp.disp(
        data['x'].values,
        data['y'].values,
        expected_src_pos_x_m,
        expected_src_pos_y_m,
        data['psi'].values
    )
    
    data['disp_dx'] = disp_dx
    data['disp_dy'] = disp_dy
    data['disp_norm'] = disp_norm
    data['disp_angle'] = disp_angle
    data['disp_sign'] = disp_sign

    return data
