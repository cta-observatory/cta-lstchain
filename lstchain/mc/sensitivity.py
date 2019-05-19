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

    e_reco = 10**events.mc_energy.to_numpy() * u.GeV

    gammaness = events.gammaness

    gevents = events[events.mc_type==0]

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

def calculate_sensitivity_lima(nex, nbg, alpha):
    """
    Sensitivity calculation using the Li & Ma formula
    eq. 17 of Li & Ma (1983).
    TODO: Caculation of the significance is INCORRECT
    by only doing a sens = 5 / significance * 100
    for the time being we leave it until implementing an
    optimized search for the 5 sigma

    Parameters
    ---------
    nex:   `float` number of excess events in the signal region
    nbg:   `float` number of events in the background region
    alpha: `float` inverse of the number of off positions

    Returns
    ---------
    sensitivity: `float` in percentage of Crab units
    """
    significance = gammapy.sensitivity(nex, nbg, alpha)
    sens = 5 / significance * 100  # percentage of Crab

    return sens


def bin_definition(gb, tb):

    max_gam = 1
    max_th2 = 0.05
    min_th2 = 0.005

    g = np.linspace(0, max_gam, gb)
    t = np.linspace(min_th2, max_th2, tb)

    return g, t


def sens(simtelfile_gammas, simtelfile_protons,
         dl2_file_g, dl2_file_p,
         nfiles_gammas, nfiles_protons,
         eb, gb, tb, noff,
         obstime = 50 * 3600 * u.s):
    """
    Main function to calculate the sensitivity given a MC dataset

    Parameters
    ---------
    simtelfile_gammas: `string` path to simtelfile of gammas with mc info
    simtelfile_protons: `string` path to simtelfile of protons with mc info
    dl2_file_g: `string` path to h5 file of reconstructed gammas
    dl2_file_p: `string' path to h5 file of reconstructed protons
    nfiles_gammas: `int` number of simtel gamma files reconstructed
    nfiles_protons: `int` number of simtel proton files reconstructed
    eb: `int` number of bins in energy
    gb: `int` number of bins in gammaness
    tb: `int` number of bins in theta2
    noff: `float` ratio between the background and the signal region
    obstime: `Quantity` Observation time in seconds

    TODO: Give files as input in a configuration file!
    Returns
    E: `array` center of energy bins
    sensitivity: `array` sensitivity per energy bin
    ---------
    """

    # Read simulated and reconstructed values
    gammaness_g, theta2_g, e_reco_g, mc_par_g = process_mc(simtelfile_gammas,
                                                                       dl2_file_g)
    gammaness_p, theta2_p, e_reco_p, mc_par_p = process_mc(simtelfile_protons,
                                                                       dl2_file_p)

    mc_par_g['sim_ev'] = mc_par_g['sim_ev']*nfiles_gammas
    mc_par_p['sim_ev'] = mc_par_p['sim_ev']*nfiles_protons

    #Pass units to GeV and cm2
    mc_par_g['emin'] = mc_par_g['emin'].to(u.GeV)
    mc_par_g['emax'] = mc_par_g['emax'].to(u.GeV)

    mc_par_p['emin'] = mc_par_p['emin'].to(u.GeV)
    mc_par_p['emax'] = mc_par_p['emax'].to(u.GeV)

    mc_par_g['area_sim'] = mc_par_g['area_sim'].to( u.cm * u.cm)
    mc_par_p['area_sim'] = mc_par_p['area_sim'].to( u.cm * u.cm)

    #Set binning for sensitivity calculation
    emin_sens = mc_par_g['emin']
    emax_sens = mc_par_g['emax']

    E = np.logspace(np.log10(emin_sens.to_value()),
                np.log10(emax_sens.to_value()), eb + 1) * u.GeV

    g, t = bin_definition(gb, tb)

    # Extract spectral parameters
    dFdE, crab_par = crab_hegra(E)
    dFdEd0, proton_par = proton_bess(E)

    # Rates and weights
    rate_g = rate(mc_par_g['emin'], mc_par_g['emax'], mc_par_g['sp_idx'],
                     mc_par_g['cone'], mc_par_g['area_sim'],
                     crab_par['f0'], crab_par['e0'])

    rate_p = rate(mc_par_p['emin'], mc_par_p['emax'], mc_par_p['sp_idx'],
                     mc_par_p['cone'], mc_par_p['area_sim'],
                     proton_par['f0'], proton_par['e0'])


    w_g = weight(mc_par_g['emin'], mc_par_g['emax'], mc_par_g['sp_idx'],
                    crab_par['alpha'], rate_g,
                    mc_par_g['sim_ev'], crab_par['e0'])

    w_p = weight(mc_par_p['emin'], mc_par_p['emax'], mc_par_p['sp_idx'],
                    proton_par['alpha'], rate_p,
                    mc_par_p['sim_ev'], proton_par['e0'])


    e_reco_gw = ((e_reco_g / crab_par['e0'])**(crab_par['alpha'] - mc_par_g['sp_idx'])) \
                * w_g
    e_reco_pw = ((e_reco_p / proton_par['e0'])**(proton_par['alpha'] - mc_par_g['sp_idx'])) \
                * w_p

    # Arrays to contain the number of gammas and hadrons for different cuts

    final_gamma = np.ndarray(shape=(eb, gb, tb))
    final_hadrons = np.ndarray(shape=(eb, gb, tb))


    #Weight events and count nº of events per bin:

    for i in range(0,eb):  # binning in energy
        for j in range(0,gb):  # cut in gammaness
            for k in range(0,tb):  # cut in theta2
                eg_w_sum = np.sum(e_reco_gw[(e_reco_g < E[i+1]) & (e_reco_g > E[i]) \
                                            & (gammaness_g > g[j]) & (theta2_g < t[k])])

                ep_w_sum = np.sum(e_reco_pw[(e_reco_p < E[i+1]) & (e_reco_p > E[i]) \
                                            & (gammaness_p > g[j]) & (theta2_p < t[k])])

                final_gamma[i][j][k] = eg_w_sum * obstime
                final_hadrons[i][j][k] = ep_w_sum * obstime

    sens = calculate_sensitivity(final_gamma, final_hadrons, 1/noff)

    #Avoid bins which are empty or have too few events:

    min_num_events = 10 #Minimum number of gamma and proton events in a bin to be taken into
    #account for minimization

    for i in range(0, eb):
        for j in range(0, gb):
            for k in range(0, tb):
                conditions = (not np.isfinite(sens[i,j,k])) or (sens[i,j,k]<=0) \
                             or (final_gamma[i,j,k] < min_num_events) \
                             or (final_hadrons[i,j,k] < min_num_events) \
                             or (not final_gamma[i,j,k] > final_hadrons[i,j,k] * 0.05)
                if conditions:
                    sens[i][j][k] = np.nan

    # Calculate the minimum sensitivity per energy bin
    sensitivity = np.ndarray(shape=eb)

    print("BEST CUTS: ")
    print("Energy bin(GeV) Gammaness Theta2(deg)")
    for i in range(0,eb):
        ind = np.unravel_index(np.argmin(sens[i], axis=None), sens[i].shape)
        print("%.2f" % E[i].to_value(), "%.2f" % g[ind[0]], "%.2f" % t[ind[1]])
        sensitivity[i] = sens[i][ind]

    return E, sensitivity
