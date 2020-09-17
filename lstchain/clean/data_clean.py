import pandas as pd

import os
import yaml
from copy import deepcopy
import numpy as np
from lstchain.io.io import dl2_params_lstcam_key
from astropy.coordinates.angle_utilities import angular_separation
import astropy.units as u

__all__ = [
        'read_and_update_dl2',
        'mc_filter',
        'data_filter'
]

def read_and_update_dl2(filepath, tel_id=1):
    """
    read DL2 files and update MC to be compliant with irf_maker
    """
    data = pd.read_hdf(filepath, key=dl2_params_lstcam_key)
    data = deepcopy(data.query(f'tel_id == {tel_id}'))

    is_simu = 'mc_type' in data.columns

    if is_simu and data.mc_type[0] >=0:
        # angles are in degrees in protopipe
        #Xi is the angular distance to the source
        data['xi'] = pd.Series(angular_separation(data.reco_az.values * u.rad,
                                                  data.reco_alt.values * u.rad,
                                                  data.mc_az.values * u.rad,
                                                  data.mc_alt.values * u.rad,
                                                  ).to(u.deg).value,
                               index=data.index)

        # FOV_offset
        data['fov_offset'] = pd.Series(angular_separation(data.reco_az.values * u.rad,
                                                      data.reco_alt.values * u.rad,
                                                      data.mc_az_tel.values * u.rad,
                                                      data.mc_alt_tel.values * u.rad,
                                                      ).to(u.deg).value,
                                   index=data.index)

    else:
        data['fov_offset'] = pd.Series(angular_separation(data.reco_az.values * u.rad,
                                                      data.reco_alt.values * u.rad,
                                                      data.az_tel.values * u.rad,
                                                      data.alt_tel.values * u.rad,
                                                      ).to(u.deg).value,
                                   index=data.index)

    return data

file = os.path.join(os.path.dirname(__file__),"../data/data_selection_cuts.yml")
with open(file, 'r') as check:
    data_cut = yaml.safe_load(check)

def mc_filter(data):

    data['pass_best_cutoff'] = data['gammaness'] > data_cut["general"]["gammaness"]
    data['pass_angular_cut'] = data['xi'] < data_cut["mc"]["xi_cut"]
    data['weight'] = np.ones(len(data))
    return data

def data_filter(data):
    if 'leakage2_intensity' in data.columns:
        data = data[data.leakage2_intensity < data_cut["data"]["leakage2_intensity"]]
    else:
        data = data[data.leakage_intensity_width_2 < data_cut["data"]["leakage_intensity_width_2"]]
    data = data[data.intensity > data_cut["data"]["intensity"]]
    data = data[data.wl > data_cut["data"]["wl"]]
    data = data[data.gammaness > data_cut["general"]["gammaness"]]
    data = data[data.fov_offset < data_cut["general"]["fov_offset"]]

    return data
