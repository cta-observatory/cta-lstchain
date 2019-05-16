import numpy as np
import pandas as pd
import astropy.units as u

from eventio.simtel.simtelfile import SimTelFile

from .plot_utils import sens_plot, sens_minimization_plot
from .mc import rate, weight

from lstchain.spectra.crab import crab_hegra
from lstchain.spectra.proton import proton_bess



__all__ = ['read_sim_par',
           'process_mc',
           'calculate_sensitivity',
           'bin_definition',
           'sens',
           ]

def read_sim_par(source):
    """
    Read MC simulated parameters

    Parameters
    ---------
    source: simtelarray file

    Returns
    ---------
    par: `dict` with simulated parameters

    """
    emin, emax = source.mc_run_headers[0]['E_range'] * u.TeV
    sp_idx = source.mc_run_headers[0]['spectral_index']
    n_showers = source.mc_run_headers[0]['n_showers']
    n_use = source.mc_run_headers[0]['n_use']
    sim_ev= n_showers * n_use
    max_impact = source.mc_run_headers[0]['core_range'][1] * u.m
    area_sim = np.pi * np.power(max_impact,2)
    cone = source.mc_run_headers[0]['viewcone'][1] * u.deg

    par_var = [emin, emax, sp_idx, sim_ev, area_sim, cone]
    par_dic = ['emin', 'emax', 'sp_idx', 'sim_ev', 'area_sim', 'cone']
    par = dict(zip(par_dic, par_var))

    return par

def process_mc(simtelfile, dl2_file):
    """
    Process the MC simulated and reconstructed to extract the relevant
    parameters to compute the sensitivity

    Paramenters
    ---------
    simtel: simtelarray file
    dl2_file: `pandas.DataFrame` dl2 parameters

    Returns
    ---------
    gammaness: `numpy.ndarray`
    theta2:    `numpy.ndarray`
    e_reco:    `numpy.ndarray` reconstructed energies
    n_reco:    `int` number of reconstructed events
    mc_par:    `dict` with simulated parameters

    """
    source = SimTelFile(simtelfile)
    sim_par = read_sim_par(source)
    events = pd.read_hdf(dl2_file)

    e_reco = 10**events.mc_energy * u.GeV
    # n_reco = e_reco.shape[0]
    gammaness = events.gammaness

    if events.iloc[0].mc_type==0:

        theta2 = (events.src_x - events.src_x_rec)**2 + \
                 (events.src_y - events.src_y_rec)**2 * u.deg**2
    else:
        theta2 = (events.src_x_rec)**2 + \
                 (events.src_y_rec)**2 * u.deg**2

    return gammaness, theta2, e_reco, sim_par


def calculate_sensitivity(nex, nbg, alpha):
    """
    Sensitivity calculation using nex/sqrt(nbg)

    Parameters
    ---------
    nex:   `float` number of excess events in the signal region
    nbg:   `float` number of events in the background region
    alpha: `float` inverse of the number of off positions

    Returns
    ---------
    sensitivity: `float` in percentage of Crab units
    """
    significance = nex / np.sqrt(nbg * alpha)
    sens = 5 / significance * 100  # percentage of Crab

    return sens

def bin_definition(gb, tb):

    max_gam = 1
    max_th2 = 0.1
    min_th2 = 0.005

    g = np.linspace(0, max_gam, gb)
    t = np.linspace(min_th2, max_th2, tb)

    return g, t

def sens(emin_sens, emax_sens, eb, gb, tb, noff, obstime = 50 * 3600 * u.s):
    """
    Main function to calculate the sensitivity given a MC dataset

    Parameters
    ---------
    eb: `int` number of bins in energy
    gb: `int` number of bins in gammaness
    tb: `int` number of bins in theta2
    noff: `float` ratio between the background and the signal region

    TODO: Give files as input in a configuration file!
    Returns
    ---------

    """

    # Read files
    simtelfile_gammas = "/home/queenmab/DATA/LST1/Gamma/gamma_20deg_0deg_run8___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel.gz"
    simtelfile_protons = "/home/queenmab/DATA/LST1/Proton/proton_20deg_0deg_run194___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel.gz"
    PATH_EVENTS = "../../cta-lstchain-extra/reco/sample_data/dl2/"
    dl2_file_g = PATH_EVENTS+"/reco_gammas.h5"
    dl2_file_p = PATH_EVENTS+"/reco_protons.h5"

    # Extract spectral parameters
    E = np.logspace(np.log10(emin_sens.to_value()),
                     np.log10(emax_sens.to_value()), eb + 1) * u.GeV

    dFdE, crab_par = crab_hegra(E)
    dFdEd0, proton_par = proton_bess(E)

    # Read simulated and reconstructed values
    gammaness_g, theta2_g, e_reco_g, mc_par_g = process_mc(simtelfile_gammas, dl2_file_g)
    gammaness_p, theta2_p, e_reco_p, mc_par_p = process_mc(simtelfile_protons, dl2_file_p)

    # Rates and weights
    rate_g = rate(mc_par_g['emin'], mc_par_g['emax'], mc_par_g['sp_idx'], \
                  mc_par_g['cone'], mc_par_g['area_sim'], crab_par['f0'], crab_par['e0'])

    rate_p = rate(mc_par_p['emin'], mc_par_p['emax'], mc_par_p['sp_idx'], \
                  mc_par_p['cone'], mc_par_p['area_sim'], proton_par['f0'], proton_par['e0'])


    w_g = weight(mc_par_g['emin'], mc_par_g['emax'], mc_par_g['sp_idx'],
                 crab_par['alpha'], rate_g, mc_par_g['sim_ev'], crab_par['e0'])

    w_p = weight(mc_par_p['emin'], mc_par_p['emax'], mc_par_p['sp_idx'],
                 proton_par['alpha'], rate_p, mc_par_p['sim_ev'], proton_par['e0'])


    e_reco_gw = ((e_reco_g / crab_par['e0'])**(crab_par['alpha'] - mc_par_g['sp_idx'])) \
        * w_g
    e_reco_pw = ((e_reco_p / proton_par['e0'])**(proton_par['alpha'] - mc_par_g['sp_idx'])) \
        * w_p

    # Arrays to contain the number of gammas and hadrons for different cuts
    final_gamma = np.ndarray(shape=(eb, gb, tb))
    final_hadrons = np.ndarray(shape=(eb, gb, tb))

    g, t = bin_definition(gb, tb)

    for i in range(0,eb):  # binning in energy
        for j in range(0,gb):  # cut in gammaness
            for k in range(0,tb):  # cut in theta2
                eg_w_sum = np.sum(e_reco_gw[(e_reco_g < E[i+1].to_value()) & (e_reco_g > E[i].to_value()) \
                                         & (gammaness_g > g[j]) & (theta2_g < t[k])])

                ep_w_sum = np.sum(e_reco_pw[(e_reco_p < E[i+1].to_value()) & (e_reco_p > E[i].to_value()) \
                                         & (gammaness_p > g[j]) & (theta2_p < t[k])])

                final_gamma[i][j][k] = eg_w_sum * obstime
                final_hadrons[i][j][k] = ep_w_sum * obstime

    sens = calculate_sensititity(final_gamma, final_hadrons, 1/noff)

    # Calculate the minimum sensitivity per energy bin
    sensitivity = np.ndarray(shape=eb)
    for i in range(0,eb):
        ind = np.unravel_index(np.argmin(sens[i], axis=None), sens[i].shape)
        sensitivity[i] = sens[i][ind]

    sens_minimization_plot(eb, gb, tb, E, sens)
    sens_plot(eb, E, sensitivity)
