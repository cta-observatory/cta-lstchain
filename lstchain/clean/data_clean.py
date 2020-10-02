import os
import json
import numpy as np

__all__ = [
        'mc_filter',
        'data_filter'
]

file = os.path.join(os.path.dirname(__file__),"../data/data_selection_cuts.json")

with open(file) as json_file:
    data_cut = json.load(json_file)

### TODO: We can generalize for the user to choose which selection cuts to make
def mc_filter(data):
    if 'leakage2_intensity' in data.columns:
        data = data[data['leakage2_intensity'] < data_cut["general"]["leakage2_intensity"]]
    else:
        data = data[data['leakage_intensity_width_2'] < data_cut["general"]["leakage_intensity_width_2"]]
    data = data[data['intensity'] > data_cut["general"]["intensity"]]
    data = data[data['wl'] > data_cut["general"]["wl"]]
    data = data[data['gh_score'] > data_cut["general"]["gh_score"]]
    data = data[data['theta'].value < data_cut["mc"]["theta_cut"]]
    data = data[data['source_fov_offset'].value < data_cut["general"]["source_fov_offset"]]

    return data

def data_filter(data):
    if 'leakage2_intensity' in data.columns:
        data = data[data['leakage2_intensity'] < data_cut["general"]["leakage2_intensity"]]
    else:
        data = data[data['leakage_intensity_width_2'] < data_cut["general"]["leakage_intensity_width_2"]]
    data = data[data['intensity'] > data_cut["general"]["intensity"]]
    data = data[data['wl'] > data_cut["general"]["wl"]]
    data = data[data['gh_score'] > data_cut["general"]["gh_score"]]
    #data = data[data['theta']] < data_cut["mc"]["theta_cut"]
    data = data[data['source_fov_offset'].value < data_cut["general"]["source_fov_offset"]]

    return data
