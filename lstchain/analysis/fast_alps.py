"""
Collection of core analysis methods
"""

import astropy.units as u
import numpy as np
import os
import pandas as pd

from astropy.coordinates import SkyCoord
from gammapy.stats import WStatCountsStatistic
from lstchain.reco.utils import compute_alpha, compute_theta2, extract_source_position, reco_source_position_sky, radec_to_camera, rotate
import fast_alps.utils.plotting as plotting
from fast_alps.utils.logger import LOGGER


def analyze_wobble(config):
    """
    Perform the wobble analysis
    """
    n_points = config['analysis']['parameters']['n_points']
    theta2_cut = config['analysis']['selection']['theta2'][0]
    LOGGER.info("Running wobble analysis with %s off-source observation points", n_points)
    LOGGER.info("Analyzing runs %s", config['analysis']['runs'])
    observation_time, data = merge_dl2_runs(config['input']['data_path'], config['analysis']['runs'])
    selected_data = apply_selection(data, config['preselection'])
    # Add theta2 to selected data
    true_source_position = extract_source_position(selected_data, config['input']['observed_source'])
    plotting.setup(config['plot_style'])
    plotting.plot_on_off(true_source_position, n_points)
    named_datasets = []
    named_datasets.append(('ON data', np.array(calculate_theta2(selected_data, true_source_position)), 1))
    n_on = np.sum(named_datasets[0][1] < theta2_cut)
    n_off = 0
    rotation_angle = 360./n_points
    origin_x = selected_data['reco_src_x']
    origin_y = selected_data['reco_src_y']
    for _ in range(1, n_points):
        t_off_data = selected_data.copy()
        off_xy = rotate(tuple(zip(origin_x, origin_y)), rotation_angle * _)
        t_off_data['reco_src_x'] = [xy[0] for xy in off_xy]
        t_off_data['reco_src_y'] = [xy[1] for xy in off_xy]
        named_datasets.append((f'OFF {rotation_angle * _}', np.array(calculate_theta2(t_off_data, true_source_position)), 1))
        n_off += np.sum(named_datasets[-1][1] < theta2_cut)

    observation_duration = pd.to_datetime(selected_data.dragon_time.iat[len(selected_data)-1], unit='s') -\
                           pd.to_datetime(selected_data.dragon_time.iat[0], unit='s')
                           
    observation_seconds = np.sum(observation_duration).total_seconds()
    stat = WStatCountsStatistic(n_on, n_off, 1./(n_points - 1))
    lima_significance = stat.significance
    lima_excess = stat.excess
    LOGGER.info('Observation time %s', observation_time)
    LOGGER.info('Number of "ON" events %s', n_on)
    LOGGER.info('Number of "OFF" events %s', n_off)
    LOGGER.info('ON/OFF observation time ratio %s', 1./(n_points - 1))
    LOGGER.info('Excess is %s', lima_excess)
    LOGGER.info('Li&Ma significance %s', lima_significance)
    plotting.plot_1d_excess(named_datasets, lima_significance, r'$\theta^2$ [deg$^2$]', theta2_cut)


def analyze_on_off(config):
    """
    Perform the ON/OFF analysis
    """
    LOGGER.info("Running ON/OFF analysis")
    LOGGER.info("ON data runs: %s", config['analysis']['runs_on'])
    observation_time_on, data_on = merge_dl2_runs(config['input']['data_path'],
                                                  config['analysis']['runs_on'])
    LOGGER.info("ON observation time: %s", observation_time_on)
    LOGGER.info("OFF data runs: %s", config['analysis']['runs_off'])
    observation_time_off, data_off = merge_dl2_runs(config['input']['data_path'],
                                                    config['analysis']['runs_off'])
    LOGGER.info("OFF observation time: %s", observation_time_off)
    #observation_time_ratio = observation_time_on / observation_time_off
    #LOGGER.info('Observation time ratio %s', observation_time_ratio)

    selected_data_on = apply_selection(data_on, config['preselection'])
    selected_data_off = apply_selection(data_off, config['preselection'])

    theta2_on = np.array(calculate_theta2(selected_data_on, (0, 0)))
    theta2_off = np.array(calculate_theta2(selected_data_off, (0, 0)))

    theta2_cut = config['analysis']['selection']['theta2'][0]
    n_on = np.sum(theta2_on < theta2_cut)
    n_off = np.sum(theta2_off < theta2_cut)
    LOGGER.info('Number of observed ON and OFF events are:\n %s, %s',
                n_on, n_off)

    theta2_norm_min = config['analysis']['selection']['theta2'][1]
    theta2_norm_max = config['analysis']['selection']['theta2'][2]
    n_norm_on = np.sum((theta2_on > theta2_norm_min) & (theta2_on < theta2_norm_max))
    n_norm_off = np.sum((theta2_off > theta2_norm_min) & (theta2_off < theta2_norm_max))
    lima_norm = n_norm_on / n_norm_off
    stat = WStatCountsStatistic(n_on, n_off, lima_norm)
    lima_significance = stat.significance
    lima_excess = stat.excess
    LOGGER.info('Excess is %s', lima_excess)
    LOGGER.info('Excess significance is %s', lima_significance)
    plotting.plot_1d_excess([('ON data', theta2_on, 1), (f'OFF data X {lima_norm:.2f}', theta2_off,  lima_norm)], lima_significance,
                            r'$\theta^2$ [deg$^2$]', theta2_cut)

    # alpha analysis
    LOGGER.info('Perform alpha analysis')
    alpha_on = np.array(calculate_alpha(selected_data_on))
    alpha_off = np.array(calculate_alpha(selected_data_off))
    alpha_cut = config['analysis']['selection']['alpha'][0]
    n_on = np.sum(alpha_on < alpha_cut)
    n_off = np.sum(alpha_off < alpha_cut)
    LOGGER.info('Number of observed ON and OFFevents are:\n %s, %s',
                n_on, n_off)

    alpha_norm_min = config['analysis']['selection']['alpha'][1]
    alpha_norm_max = config['analysis']['selection']['alpha'][2]
    n_norm_on = np.sum((alpha_on > alpha_norm_min) & (alpha_on < alpha_norm_max))
    n_norm_off = np.sum((alpha_off > alpha_norm_min) & (alpha_off < alpha_norm_max))
    lima_norm = n_norm_on / n_norm_off
    stat = WStatCountsStatistic(n_on, n_off, lima_norm)
    lima_significance = stat.significance
    lima_excess = stat.excess
    LOGGER.info('Excess is %s', lima_excess)
    LOGGER.info('Excess significance is %s', lima_significance)
    plotting.plot_1d_excess([('ON data', alpha_on, 1), (f'OFF data X {lima_norm:.2f}', alpha_off,  lima_norm)], lima_significance,
                            r'$\alpha$ [deg]', alpha_cut, 0, 90, 90)
