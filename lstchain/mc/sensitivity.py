import numpy as np
import pandas as pd
import astropy.units as u
import math
from eventio.simtel.simtelfile import SimTelFile
from .plot_utils import sens_plot, sens_minimization_plot
from .mc import rate, weight
from lstchain.spectra.crab import crab_hegra
from lstchain.spectra.proton import proton_bess
from gammapy.stats.poisson import excess_matching_significance_on_off
from lstchain.reco.utils import reco_source_position_sky
from  astropy.coordinates.angle_utilities import angular_separation
from astropy.coordinates import SkyCoord

__all__ = ['read_sim_par',
           'process_mc',
           'calculate_sensitivity',
           'calculate_sensitivity_lima',
           'bin_definition',
           'ring_containment',
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

def process_mc(simtelfile, dl2_file, mc_type):
    """
    Process the MC simulated and reconstructed to extract the relevant
    parameters to compute the sensitivity

    Paramenters
    ---------
    simtel: simtelarray file
    dl2_file: `pandas.DataFrame` dl2 parameters
    mc_type: 'string' type of particle

    Returns
    ---------
    gammaness: `numpy.ndarray`
    angdist2:  `numpy.ndarray` angular distance squared
    e_reco:    `numpy.ndarray` reconstructed energies
    n_reco:    `int` number of reconstructed events
    mc_par:    `dict` with simulated parameters

    """
    source = SimTelFile(simtelfile)
    sim_par = read_sim_par(source)
    events = pd.read_hdf(dl2_file)

    e_reco = 10**events.mc_energy.to_numpy() * u.GeV
    gammaness = events.gammaness

    #Get source position in radians

    focal_length = source.telescope_descriptions[1]['camera_settings']['focal_length'] * u.m


    # If the particle is a gamma ray, it returns the squared angular distance
    # from the reconstructed gamma-ray position and the simulated incoming position
    if mc_type=='gamma':
        events = events[events.mc_type==0]
        alt2 = events.mc_alt
        az2 = np.arctan(np.tan(events.mc_az))

    # If the particle is not a gamma-ray (diffuse protons/electrons), it returns
    # the squared angular distance of the reconstructed position w.r.t. the
    # center of the camera
    else:
        events = events[events.mc_type!=0]
        alt2 = events.mc_alt_tel
        az2 = np.arctan(np.tan(events.mc_az_tel))

    src_pos_reco = reco_source_position_sky(events.x.values * u.m,
                                            events.y.values * u.m,
                                            events.disp_dx_rec.values * u.m,
                                            events.disp_dy_rec.values * u.m,
                                            focal_length,
                                            events.mc_alt_tel.values * u.rad,
                                            events.mc_az_tel.values * u.rad)

    alt1 = src_pos_reco.alt.rad
    az1 = np.arctan(np.tan(src_pos_reco.az.rad))

    angdist2 = (angular_separation(az1, alt1, az2, alt2).to_numpy() * u.rad)**2
    return gammaness, angdist2.to(u.deg**2), e_reco, sim_par


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
    sens: `float` in percentage of Crab units
    """
    significance = nex / np.sqrt(nbg * alpha)
    sens = 5 / significance * 100  # percentage of Crab

    return sens

def calculate_sensitivity_lima(nex, nbg, alpha):
    """
    Sensitivity calculation using the Li & Ma formula
    eq. 17 of Li & Ma (1983).
    https://ui.adsabs.harvard.edu/abs/1983ApJ...272..317L/abstract

    Parameters
    ---------
    nex:   `float` number of excess events in the signal region
    nbg:   `float` number of events in the background region
    alpha: `float` inverse of the number of off positions

    Returns
    ---------
    sens: `float` in percentage of Crab units
    """
    nex_5sigma = excess_matching_significance_on_off(\
        n_off=nbg,alpha=alpha,significance=5,method='lima')
    sens = nex_5sigma / nex * 100  # percentage of Crab

    return sens


def bin_definition(gb, tb):
    """
    Define binning in gammaness and theta2 for the
    optimization of the sensitivity

    Parameters
    ---------
    gb:   `int` number of bins in gammaness
    tb:   `int` number of bins in theta2

    Returns
    ---------
    g, t: `numpy.ndarray` binning of gammaness and theta2
    """
    max_gam = 1
    max_th2 = 0.05 * u.deg * u.deg
    min_th2 = 0.005 * u.deg * u.deg

    g = np.linspace(0, max_gam, gb)
    t = np.linspace(min_th2, max_th2, tb)

    return g, t

def ring_containment(angdist2, ring_radius, ring_halfwidth):
    """
    Calculate containment of cosmic ray particles with reconstructed positions
    within a ring of radius=ring_radius and half width=ring_halfwidth
    Parameters
    ---------
    angdist2:       `numpy.ndarray` angular distance squared w.r.t.
                    the center of the camera
    ring_radius:    `float` ring radius
    ring_halfwidth: `float` halfwidth of the ring

    Returns
    ---------
    contained: `numpy.ndarray` bool array
    area: angular area of the ring
    """
    ring_lower_limit = ring_radius - ring_halfwidth
    ring_upper_limit = np.sqrt(2 * (ring_radius**2) - (ring_lower_limit)**2)

    area = np.pi * (ring_upper_limit**2 - ring_lower_limit**2)
    # For the two halfwidths to cover the same area, compute the area of
    # the internal and external rings:
    # A_internal = pi * ((ring_radius**2) - (ring_lower_limit)**2)
    # A_external = pi * ((ring_upper_limit**2) - (ring_radius)**2)
    # The areas should be equal, so we can extract the ring_upper_limit
    # ring_upper_limit = math.sqrt(2 * (ring_radius**2) - (ring_lower_limit)**2)

    contained = np.where((np.sqrt(angdist2) < ring_upper_limit) & (np.sqrt(angdist2) > ring_lower_limit), True, False)

    return contained, area

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
                                                           dl2_file_g, 'gamma')
    gammaness_p, angdist2_p, e_reco_p, mc_par_p = process_mc(simtelfile_protons,
                                                             dl2_file_p, 'proton')

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
    emin_sens = 10**1 * u.GeV #mc_par_g['emin']
    emax_sens = 10**5 * u.GeV #mc_par_g['emax']

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

    p_contained, ang_area_p = ring_containment(angdist2_p, 0.4 * u.deg, 0.1 * u.deg)
    # FIX: ring_radius and ring_halfwidth should have units of deg
    # FIX: hardcoded at the moment, but ring_radius should be read from
    # the gamma file (point-like) or given as input (diffuse).
    # FIX: ring_halfwidth should be given as input
    area_ratio_p = np.pi * t / ang_area_p
    # ratio between the area where we search for protons ang_area_p
    # and the area where we search for gammas math.pi * t

    # Arrays to contain the number of gammas and hadrons for different cuts
    final_gamma = np.ndarray(shape=(eb, gb, tb))
    final_hadrons = np.ndarray(shape=(eb, gb, tb))


    # Weight events and count number of events per bin:
    for i in range(0,eb):  # binning in energy
        for j in range(0,gb):  # cut in gammaness
            for k in range(0,tb):  # cut in theta2
                eg_w_sum = np.sum(e_reco_gw[(e_reco_g < E[i+1]) & (e_reco_g > E[i]) \
                                            & (gammaness_g > g[j]) & (theta2_g < t[k])])

                ep_w_sum = np.sum(e_reco_pw[(e_reco_p < E[i+1]) & (e_reco_p > E[i]) \
                                            & (gammaness_p > g[j]) & p_contained])

                final_gamma[i][j][k] = eg_w_sum * obstime
                final_hadrons[i][j][k] = ep_w_sum * obstime * area_ratio_p[k]

    sens = calculate_sensitivity_lima(final_gamma, final_hadrons * noff, 1/noff)

    # Avoid bins which are empty or have too few events:
    min_num_events = 10
    # Minimum number of gamma and proton events in a bin to be taken into account for minimization

    for i in range(0, eb):
        for j in range(0, gb):
            for k in range(0, tb):
                conditions = (not np.isfinite(sens[i,j,k])) or (sens[i,j,k]<=0) \
                             or (final_gamma[i,j,k] < min_num_events) \
                             or (final_hadrons[i,j,k] < min_num_events) \
                             or (not final_gamma[i,j,k] > final_hadrons[i,j,k] * 0.05)
                if conditions:
                    sens[i][j][k] = np.inf

    # Calculate the minimum sensitivity per energy bin
    sensitivity = np.ndarray(shape=eb)

    print("BEST CUTS: ")
    print("Energy bin(GeV) Gammaness Theta2(deg2) Ngamma Nbkg Ngamma/min Nbkg/min")
    for i in range(0,eb):
        ind = np.unravel_index(np.nanargmin(sens[i], axis=None), sens[i].shape)
        print("%.2f" % E[i].to_value(),"-","%.2f" % E[i+1].to_value(),"%.2f" % g[ind[0]],
              "%.2f" % t[ind[1]].to_value(), "%.2f" % final_gamma[i][ind],
              "%.2f" % final_hadrons[i][ind], "%.2f" % (final_gamma[i][ind]/(60*50)),
              "%.2f" % (final_hadrons[i][ind]/(60*50)))
        sensitivity[i] = sens[i][ind]

    return E, sensitivity
