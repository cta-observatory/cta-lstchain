"""
Collection of core analysis methods
"""

import getpass
import logging
import numpy as np
import matplotlib.pyplot as plt
import time

from gammapy.stats import WStatCountsStatistic
from lstchain.io.io import merge_dl2_runs
from lstchain.reco.utils import compute_alpha, compute_theta2, extract_source_position, filter_events, rotate
import lstchain.visualization.plot_dl2 as plotting


LOGGER = logging.getLogger('post_dl2')
LOGGER.setLevel(logging.DEBUG)
LOGGING_LEVELS = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}


def setup_logging(verbosity=1):
    """
    Setup logger console and file descriptors

    Two log stream handlers are added, one for file-based logging and one for console output.
    Logging level to file is always set to DEBUG and console verbosity can be controlled.
    Verbosity levels {0,1,2} correspond to {ERROR, INFO, DEBUG}.

    :param int verbosity: Verbosity level used for console output
    """
    fh = logging.FileHandler(f'/tmp/lstchain-postDL2-{getpass.getuser()}_{time.time()}.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(LOGGING_LEVELS[verbosity])
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    LOGGER.addHandler(console)
    LOGGER.addHandler(fh)


def analyze_wobble(config):
    """
    Extracts the theta2 plot of a dataset taken with wobble observations
    
    Parameters
    ----------
    config_file

    """
   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    n_points = config['analysis']['parameters']['n_points']
    theta2_cut = config['analysis']['selection']['theta2'][0]
    LOGGER.info("Running wobble analysis with %s off-source observation points", n_points)
    LOGGER.info("Analyzing runs %s", config['analysis']['runs'])
    observation_time, data = merge_dl2_runs(config['input']['data_tag'],
                                            config['analysis']['runs'],
                                            config['input']['columns_to_read'])
    LOGGER.debug('\nPreselection:\n%s',config['preselection'])
    for key, value in config['preselection'].items():
        LOGGER.debug('\nParameter: %s, range: %s, value type: %s', key, value, type(value))

    selected_data = filter_events(data, config['preselection'])
    # Add theta2 to selected data
    true_source_position = extract_source_position(selected_data, config['input']['observed_source'])
    plotting.plot_wobble(true_source_position, n_points, ax1)
    named_datasets = []
    named_datasets.append(('ON data', np.array(compute_theta2(selected_data, true_source_position)), 1))
    n_on = np.sum(named_datasets[0][1] < theta2_cut)
    n_off = 0
    rotation_angle = 360./n_points
    origin_x = selected_data['reco_src_x']
    origin_y = selected_data['reco_src_y']
    for off_point in range(1, n_points):
        t_off_data = selected_data.copy()
        off_xy = rotate(tuple(zip(origin_x, origin_y)), rotation_angle * off_point)
        t_off_data['reco_src_x'] = [xy[0] for xy in off_xy]
        t_off_data['reco_src_y'] = [xy[1] for xy in off_xy]
        named_datasets.append((f'OFF {rotation_angle * off_point}', np.array(compute_theta2(t_off_data, true_source_position)), 1))
        n_off += np.sum(named_datasets[-1][1] < theta2_cut)

    stat = WStatCountsStatistic(n_on, n_off, 1./(n_points - 1))

    # API change for attributes significance and excess in the new gammapy version: https://docs.gammapy.org/dev/api/gammapy.stats.WStatCountsStatistic.html
    lima_significance = stat.sqrt_ts.item()
    lima_excess = stat.n_sig
    LOGGER.info('Observation time %s', observation_time)
    LOGGER.info('Number of "ON" events %s', n_on)
    LOGGER.info('Number of "OFF" events %s', n_off)
    LOGGER.info('ON/OFF observation time ratio %s', 1./(n_points - 1))
    LOGGER.info('Excess is %s', lima_excess)
    LOGGER.info('Li&Ma significance %s', lima_significance)
    plotting.plot_1d_excess(named_datasets, lima_significance, r'$\theta^2$ [deg$^2$]', theta2_cut, ax2)

    if config['output']['interactive'] is True:
        LOGGER.info('Interactive mode ON, plots will be only shown, but not saved')
        plt.show()
    else:
        LOGGER.info('Interactive mode OFF, no plots will be displayed')
        plt.ioff()
        plt.savefig(f"{config['output']['directory']}/wobble.png")
        plt.close()


def analyze_on_off(config):
    """
    Extracts the theta2 plot of a dataset taken with ON/OFF observations
    
    Parameters
    ----------
    config_file

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    LOGGER.info("Running ON/OFF analysis")
    LOGGER.info("ON data runs: %s", config['analysis']['runs_on'])
    observation_time_on, data_on = merge_dl2_runs(config['input']['data_tag'],
                                                  config['analysis']['runs_on'],
                                                  config['input']['columns_to_read'], 4)
    LOGGER.info("ON observation time: %s", observation_time_on)
    LOGGER.info("OFF data runs: %s", config['analysis']['runs_off'])
    observation_time_off, data_off = merge_dl2_runs(config['input']['data_tag'],
                                                    config['analysis']['runs_off'],
                                                    config['input']['columns_to_read'], 4)
    LOGGER.info("OFF observation time: %s", observation_time_off)
    #observation_time_ratio = observation_time_on / observation_time_off
    #LOGGER.info('Observation time ratio %s', observation_time_ratio)

    selected_data_on = filter_events(data_on, config['preselection'])
    selected_data_off = filter_events(data_off, config['preselection'])

    theta2_on = np.array(compute_theta2(selected_data_on, (0, 0)))
    theta2_off = np.array(compute_theta2(selected_data_off, (0, 0)))

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
    lima_significance = stat.sqrt_ts.item()
    lima_excess = stat.n_sig
    LOGGER.info('Excess is %s', lima_excess)
    LOGGER.info('Excess significance is %s', lima_significance)
    plotting.plot_1d_excess([('ON data', theta2_on, 1), (f'OFF data X {lima_norm:.2f}', theta2_off,  lima_norm)], lima_significance,
                            r'$\theta^2$ [deg$^2$]', theta2_cut, ax1)

    # alpha analysis
    LOGGER.info('Perform alpha analysis')
    alpha_on = np.array(compute_alpha(selected_data_on))
    alpha_off = np.array(compute_alpha(selected_data_off))
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
    lima_significance = stat.sqrt_ts.item()
    lima_excess = stat.n_sig
    LOGGER.info('Excess is %s', lima_excess)
    LOGGER.info('Excess significance is %s', lima_significance)
    plotting.plot_1d_excess([('ON data', alpha_on, 1), (f'OFF data X {lima_norm:.2f}', alpha_off,  lima_norm)], lima_significance,
                            r'$\alpha$ [deg]', alpha_cut, ax2, 0, 90, 90)
    if config['output']['interactive'] is True:
        LOGGER.info('Interactive mode ON, plots will be only shown, but not saved')
        plt.show()
    else:
        LOGGER.info('Interactive mode OFF, no plots will be displayed')
        plt.ioff()
        plt.savefig(f"{config['output']['directory']}/on_off.png")
        plt.close()
