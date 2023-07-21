import numpy as np
import pandas as pd

import astropy.units as u
from lstchain.io.io import dl2_params_lstcam_key

from lstchain.mc.sensitivity import (
    process_mc,
    process_real,
    read_sim_par,
    calculate_sensitivity,
    calculate_sensitivity_lima,
    bin_definition,
    ring_containment,
    diff_events_after_cut_real,
    diff_events_after_cut,
    samesign,
    find_cut,
    find_cut_real
)

def test_read_sim_par(simulated_dl1_file):
    par = read_sim_par(simulated_dl1_file)

    assert u.isclose(par['emin'], 0.1 * u.TeV)
    assert u.isclose(par['emax'], 5.0 * u.TeV)
    assert np.isclose(par['sp_idx'], -2.0)
    assert np.isclose(par['sim_ev'], 100)
    assert u.isclose(par['area_sim'], np.pi * (250*u.m)**2)
    assert u.isclose(par['cone'], 2 * u.deg)


def test_process_mc(simulated_dl2_file):
    #Write some delta_t and dragon_time in the example dl2 file
    dl2 = pd.read_hdf(simulated_dl2_file, key=dl2_params_lstcam_key)
    dl2["delta_t"]=0.0005
    timestamps = np.random.uniform(0, 600, len(dl2))
    timestamps.sort()
    dl2["dragon_time"]=timestamps
    dl2.to_hdf(simulated_dl2_file, key=dl2_params_lstcam_key)
    process_mc(simulated_dl2_file, 'gamma')
    process_mc(simulated_dl2_file, 'proton')
    process_real(simulated_dl2_file)


def test_diff_events_after_cut(simulated_dl2_file):

    events=pd.read_hdf(simulated_dl2_file, key=dl2_params_lstcam_key)
    events["theta2"]=0.01
    diff_events_after_cut_real(events, events, 10, 10, 'gammaness', 0.5, 0.5)
    diff_events_after_cut_real(events, events, 10, 10, 'theta2', 0.5, 0.5)
    diff_events_after_cut(events, np.ones(len(events)), 10, 'gammaness', 0.5, 0.5)
    diff_events_after_cut(events, np.ones(len(events)), 10, 'theta2', 0.5, 0.5)

def test_find_cut(simulated_dl2_file):
    events=pd.read_hdf(simulated_dl2_file, key=dl2_params_lstcam_key)
    events["theta2"]=0.01
    find_cut(events, np.ones(len(events)), 10, 'gammaness', 0.5, 0.5, 0.5)
    find_cut(events, np.ones(len(events)), 10, 'theta2', 0.5, 0.5, 0.5)
    find_cut_real(events, events, 10, 10, 'gammaness', 0.5, 0.5, 0.5)
    find_cut_real(events, events, 10, 10, 'theta2', 0.5, 0.5, 0.5)

def test_samesign():
    a=1
    b=-1
    assert samesign(a,b)==False

def test_calculate_sensitivity():
    np.testing.assert_allclose(calculate_sensitivity(
        50, 10, 0.2), 14.14, rtol=1.e-3)
    np.testing.assert_allclose(calculate_sensitivity(
        200, 50, 1), 17.67, rtol=1.e-3)
    # Testing an array
    np.testing.assert_allclose(calculate_sensitivity(
        [10, 100], [50, 100], 1), [353.55, 50.], rtol=1.e-3)


def test_calculate_sensitivity_lima():
    # Testing an array
    np.testing.assert_allclose(calculate_sensitivity_lima(
            np.array([10, 100]), np.array([50,100]), np.array([1, 1])),
                               (np.array([63.00, 83.57]), np.array([630.07,  83.57])), rtol = 1.e-3)

def test_bin_definition():
    gammaness_bins, theta2_bins = bin_definition(10,10)
    gammaness_bins_test = [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    theta2_bins_test = [0.005, 0.01 , 0.015, 0.02 , 0.025, 0.03 , 0.035, 0.04,
            0.045, 0.05]
    np.testing.assert_allclose(gammaness_bins, gammaness_bins_test)
    np.testing.assert_allclose(theta2_bins.to_value(), theta2_bins_test)

def test_ring_containment():
    # Event contained
    contained, area = ring_containment(0.4 ** 2, 0.5, 0.101)
    np.testing.assert_allclose(area, 0.57, rtol=1.e-3)
    np.testing.assert_equal(contained, True)

    # Event not contained
    contained, area = ring_containment(0.1 ** 2, 0.4, 0.1)
    np.testing.assert_allclose(area, 0.44, rtol=1.e-3)
    np.testing.assert_equal(contained, False)

    # Testing an array
    contained, area = ring_containment(np.linspace(0.1, 1, 10), 0.6, 0.2)
    np.testing.assert_allclose(area, 1.256, rtol=1.e-3)
    np.testing.assert_equal(contained, [False, True, True, True, True,
                                        False, False, False, False, False])

def test_sensitivity(simulated_dl2_file):
    from lstchain.mc.sensitivity import (sensitivity_gamma_efficiency,
                                         sensitivity_gamma_efficiency_real_protons,
                                         sensitivity_gamma_efficiency_real_data)
    import astropy.units as u


    #Write some delta_t and timestamps in the example dl2 file
    dl2 = pd.read_hdf(simulated_dl2_file, key=dl2_params_lstcam_key)
    dl2["delta_t"]=0.0005
    timestamps = np.random.uniform(0, 600, len(dl2))
    timestamps.sort()
    dl2["dragon_time"]=timestamps
    dl2.to_hdf(simulated_dl2_file, key=dl2_params_lstcam_key)

    geff_gammaness = 0.9
    geff_theta2 = 0.8
    eb = 10
    obstime = 50 * 3600 * u.s
    noff = 2

    sensitivity_gamma_efficiency(simulated_dl2_file,
                                 simulated_dl2_file,
                                 1, 1,
                                 eb,
                                 geff_gammaness,
                                 geff_theta2,
                                 noff,
                                 obstime)

    sensitivity_gamma_efficiency_real_protons(simulated_dl2_file,
                                              simulated_dl2_file,
                                              1,
                                              eb,
                                              geff_gammaness,
                                              geff_theta2,
                                              noff,
                                              obstime)

    sensitivity_gamma_efficiency_real_data(simulated_dl2_file,
                                           simulated_dl2_file,
                                           np.zeros(eb),np.ones(eb),
                                           eb, np.ones(eb+1) * u.TeV,
                                           geff_gammaness,
                                           geff_theta2,
                                           noff,
                                           obstime)
