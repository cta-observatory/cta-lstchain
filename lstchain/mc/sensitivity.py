import numpy as np
import pandas as pd
import astropy.units as u
from .plot_utils import sensitivity_minimization_plot, plot_positions_survived_events
from .mc import rate, weight
from lstchain.spectra.crab import crab_hegra,crab_magic
from lstchain.spectra.proton import proton_bess
from gammapy.stats import WStatCountsStatistic
from lstchain.reco.utils import reco_source_position_sky
from astropy.coordinates.angle_utilities import angular_separation
from lstchain.io import read_simu_info_merged_hdf5
from lstchain.io.io import dl2_params_lstcam_key

__all__ = [
    'read_sim_par',
    'process_mc',
    'calculate_sensitivity',
    'calculate_sensitivity_lima',
    'calculate_sensitivity_lima_ebin',
    'bin_definition',
    'ring_containment',
    'find_best_cuts_sensitivity',
    'sensitivity',
    ]


def read_sim_par(dl1_file):
    """
    Read MC simulated parameters

    Parameters
    ---------
    dl1_file: `pandas.DataFrame` dl1 parameters

    Returns
    ---------
    par: `dict` with simulated parameters

    """
    simu_info = read_simu_info_merged_hdf5(dl1_file)
    emin = simu_info.energy_range_min
    emax = simu_info.energy_range_max
    sp_idx = simu_info.spectral_index
    sim_ev = simu_info.num_showers * simu_info.shower_reuse
    area_sim = (simu_info.max_scatter_range - simu_info.min_scatter_range) ** 2 * np.pi
    cone = simu_info.max_viewcone_radius

    par_var = [emin, emax, sp_idx, sim_ev, area_sim, cone]
    par_dic = ['emin', 'emax', 'sp_idx', 'sim_ev', 'area_sim', 'cone']
    par = dict(zip(par_dic, par_var))

    return par


def process_mc(dl1_file, dl2_file, mc_type):
    """
    Process the MC simulated and reconstructed to extract the relevant
    parameters to compute the sensitivity

    Paramenters
    ---------
    dl1_file: `pandas.DataFrame` dl1 parameters
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
    sim_par = read_sim_par(dl1_file)

    events = pd.read_hdf(dl2_file, key = dl2_params_lstcam_key)

    # Filters:
    # TO DO: These cuts must be given in a configuration file
    # By now: only cut in leakage and intensity
    # we use all telescopes (number of events needs to be multiplied 
    # by the number of LSTs in the simulation)

    filter_good_events = (
        (events.leakage_intensity_width_2 < 0.2)
        & (events.intensity > 50)
    )


    events = events[filter_good_events]

    e_reco = events.reco_energy.to_numpy() * u.TeV
    e_true = events.mc_energy.to_numpy() * u.TeV

    gammaness = events.gammaness

    # TO DO: Focal length should be read from file
    #focal_length = source.telescope_descriptions[1]['camera_settings']['focal_length'] * u.m
    focal_length = 28 * u.m

    # If the particle is a gamma ray, it returns the squared angular distance
    # from the reconstructed gamma-ray position and the simulated incoming position
    if mc_type == 'gamma':
        events = events[events.mc_type == 0]
        alt2 = events.mc_alt
        az2 = np.arctan(np.tan(events.mc_az))

    # If the particle is not a gamma-ray (diffuse protons/electrons), it returns
    # the squared angular distance of the reconstructed position w.r.t. the
    # center of the camera
    else:
        events = events[events.mc_type != 0]
        alt2 = events.mc_alt_tel
        az2 = np.arctan(np.tan(events.mc_az_tel))

    # Remember events['src_x'] and events['src_x_rec'] are in meters
    src_pos_reco = reco_source_position_sky(events.x.values * u.m,
                                            events.y.values * u.m,
                                            events.reco_disp_dx.values * u.m,
                                            events.reco_disp_dy.values * u.m,
                                            focal_length,
                                            events.mc_alt_tel.values * u.rad,
                                            events.mc_az_tel.values * u.rad)

    alt1 = src_pos_reco.alt.rad
    az1 = np.arctan(np.tan(src_pos_reco.az.rad))

    angdist2 = (angular_separation(az1, alt1, az2, alt2).to_numpy() * u.rad) ** 2
    events['theta2'] = angdist2

    return gammaness, angdist2.to(u.deg ** 2), e_reco, e_true, sim_par, events


def calculate_sensitivity(n_excesses, n_background, alpha):
    """
    Sensitivity calculation using n_excesses/sqrt(n_background)

    Parameters
    ---------
    n_excesses:   `numpy.ndarray` number of excess events in the signal region
    n_background:   `numpy.ndarray` number of events in the background region
    alpha: `numpy.ndarray` inverse of the number of off positions

    Returns
    ---------
    sensitivity: `numpy.ndarray` in percentage of Crab units
    """
    significance = n_excesses / np.sqrt(n_background * alpha)
    sensitivity = 5 / significance * 100  # percentage of Crab

    return sensitivity

def calculate_sensitivity_lima(n_on_events, n_background, alpha, n_bins_energy, n_bins_gammaness, n_bins_theta2):
    """
    Sensitivity calculation using the Li & Ma formula
    eq. 17 of Li & Ma (1983).
    https://ui.adsabs.harvard.edu/abs/1983ApJ...272..317L/abstract

    We calculate the sensitivity in bins of energy, gammaness and
    theta2

    Parameters
    ---------
    n_on_events:   `numpy.ndarray` number of ON events in the signal region
    n_background:   `numpy.ndarray` number of events in the background region
    alpha: `float` inverse of the number of off positions
    n_bins_energy: `int` number of bins in energy
    n_bins_gammaness: `int` number of bins in gammaness
    n_bins_theta2: `int` number of bins in theta2

    Returns
    ---------
    sensitivity: `numpy.ndarray` sensitivity in percentage of Crab units
    n_excesses_5sigma: `numpy.ndarray` number of excesses corresponding to 
                a 5 sigma significance

    """

    
    stat = WStatCountsStatistic(
        n_on=n_on_events,  
        n_off=n_background,
        alpha=alpha
        )


    n_excesses_5sigma = stat.excess_matching_significance(5)

    for i in range(0, n_bins_energy):
        for j in range(0, n_bins_gammaness):
            for k in range(0, n_bins_theta2):
                if n_excesses_5sigma[i][j][k] < 10:
                    n_excesses_5sigma[i][j][k] = 10

                if n_excesses_5sigma[i, j, k] < 0.05 * n_background[i][j][k] / 5:
                    n_excesses_5sigma[i, j, k] = 0.05 * n_background[i][j][k] / 5


    sensitivity = n_excesses_5sigma / n_on_events * 100  # percentage of Crab

    return n_excesses_5sigma, sensitivity


def calculate_sensitivity_lima_ebin(n_on_events, n_background, alpha, n_bins_energy):
    """
    Sensitivity calculation using the Li & Ma formula
    eq. 17 of Li & Ma (1983).
    https://ui.adsabs.harvard.edu/abs/1983ApJ...272..317L/abstract

    Parameters
    ---------
    n_on_events:   `numpy.ndarray` number of ON events in the signal region
    n_background: `numpy.ndarray` number of events in the background region
    alpha:        `float` inverse of the number of off positions
    n_bins_energy:`int` number of bins in energy

    Returns
    ---------
    sensitivity: `numpy.ndarray` sensitivity in percentage of Crab units
    n_excesses_5sigma: `numpy.ndarray` number of excesses corresponding to 
                a 5 sigma significance

    """
    #if any(len(a) != n_bins_energy for a in (n_on_events, n_background, alpha)):
    #    raise ValueError(
     #       'Excess, background and alpha arrays must have the same length')

    stat = WStatCountsStatistic(
        n_on=n_on_events,
        n_off=n_background,
        alpha=alpha 
        )
        


    n_excesses_5sigma = stat.excess_matching_significance(5)

    for i in range(0, n_bins_energy):
        # If the excess needed to get 5 sigma is less than 10,
        # we force it to be at least 10
        if n_excesses_5sigma[i] < 10:
            n_excesses_5sigma[i] = 10
        # If the excess needed to get 5 sigma is less than 5%
        # of the background, we force it to be at least 5% of
        # the background
        if n_excesses_5sigma[i] < 0.05 * n_background[i] * alpha[i]:
            n_excesses_5sigma[i] = 0.05 * n_background[i] * alpha[i]

    sensitivity = n_excesses_5sigma / n_on_events * 100  # percentage of Crab

    return n_excesses_5sigma, sensitivity

def bin_definition(n_bins_gammaness, n_bins_theta2):
    """
    Define binning in gammaness and theta2 for the
    optimization of the sensitivity

    Parameters
    ---------
    n_bins_gammaness:   `int` number of bins in gammaness
    n_bins_theta2:   `int` number of bins in theta2

    Returns
    ---------
    gammaness_bins, theta2_bins: `numpy.ndarray` binning of gammaness and theta2

    """
    max_gam = 1
    max_th2 = 0.05 * u.deg * u.deg
    min_th2 = 0.005 * u.deg * u.deg

    gammaness_bins = np.linspace(0, max_gam, n_bins_gammaness)
    theta2_bins = np.linspace(min_th2, max_th2, n_bins_theta2)

    return gammaness_bins, theta2_bins


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
    ring_upper_limit = np.sqrt(2 * (ring_radius ** 2) - (ring_lower_limit) ** 2)

    area = np.pi * (ring_upper_limit ** 2 - ring_lower_limit ** 2)
    # For the two halfwidths to cover the same area, compute the area of
    # the internal and external rings:
    # A_internal = pi * ((ring_radius**2) - (ring_lower_limit)**2)
    # A_external = pi * ((ring_upper_limit**2) - (ring_radius)**2)
    # The areas should be equal, so we can extract the ring_upper_limit
    # ring_upper_limit = math.sqrt(2 * (ring_radius**2) - (ring_lower_limit)**2)

    contained = np.where((np.sqrt(angdist2) < ring_upper_limit) & (np.sqrt(angdist2) > ring_lower_limit), True, False)

    return contained, area

def find_best_cuts_sensitivity(simtelfile_gammas, simtelfile_protons,
                               dl2_file_g, dl2_file_p,
                               nfiles_gammas, nfiles_protons,
                               n_bins_energy, n_bins_gammaness, n_bins_theta2, noff,
                               obstime = 50 * 3600 * u.s):

    """
    Main function to find the best cuts to calculate the sensitivity
    based on a given a MC data subset

    Parameters
    ---------
    simtelfile_gammas: `string` path to simtelfile of gammas with mc info
    simtelfile_protons: `string` path to simtelfile of protons with mc info
    dl2_file_g: `string` path to h5 file of reconstructed gammas
    dl2_file_p: `string' path to h5 file of reconstructed protons
    nfiles_gammas: `int` number of simtel gamma files reconstructed
    nfiles_protons: `int` number of simtel proton files reconstructed
    n_bins_energy: `int` number of bins in energy
    n_bins_gammaness: `int` number of bins in gammaness
    n_bins_theta2: `int` number of bins in theta2
    noff: `float` ratio between the background and the signal region
    obstime: `Quantity` Observation time in seconds

    Returns
    ---------
    energy: `array` center of energy bins
    sensitivity: `array` sensitivity per energy bin

    """

    # Read simulated and reconstructed values
    gammaness_g, theta2_g, e_reco_g, e_true_g, mc_par_g, events_g = process_mc(simtelfile_gammas,
                                                                               dl2_file_g, 'gamma')
    gammaness_p, angdist2_p, e_reco_p, e_true_p, mc_par_p, events_p = process_mc(simtelfile_protons,
                                                                                 dl2_file_p, 'proton')

    mc_par_g['sim_ev'] = mc_par_g['sim_ev'] * nfiles_gammas
    mc_par_p['sim_ev'] = mc_par_p['sim_ev'] * nfiles_protons

    # Pass units to TeV and cm2
    mc_par_g['emin'] = mc_par_g['emin'].to(u.TeV)
    mc_par_g['emax'] = mc_par_g['emax'].to(u.TeV)

    mc_par_p['emin'] = mc_par_p['emin'].to(u.TeV)
    mc_par_p['emax'] = mc_par_p['emax'].to(u.TeV)

    mc_par_g['area_sim'] = mc_par_g['area_sim'].to(u.cm ** 2)
    mc_par_p['area_sim'] = mc_par_p['area_sim'].to(u.cm ** 2)

    # Set binning for sensitivity calculation
    # TODO: This information should be read from the files
    emin_sensitivity = 0.01 * u.TeV  # mc_par_g['emin']
    emax_sensitivity =  100 * u.TeV  # mc_par_g['emax']

    energy = np.logspace(np.log10(emin_sensitivity.to_value()),
                         np.log10(emax_sensitivity.to_value()), n_bins_energy + 1) * u.TeV

    gammaness_bins, theta2_bins = bin_definition(n_bins_gammaness, n_bins_theta2)

    # Extract spectral parameters
    dFdE, crab_par = crab_hegra(energy)
    dFdEd0, proton_par = proton_bess(energy)

    bins = np.logspace(np.log10(emin_sensitivity.to_value()), np.log10(emax_sensitivity.to_value()), n_bins_energy + 1)

    y0 = mc_par_g['sim_ev'] / (mc_par_g['emax'].to_value()**(mc_par_g['sp_idx'] + 1) \
                               - mc_par_g['emin'].to_value()**(mc_par_g['sp_idx'] + 1)) \
        * (mc_par_g['sp_idx'] + 1)
    y = y0 * (bins[1:]**(crab_par['alpha'] + 1) - bins[:-1]**(crab_par['alpha'] + 1)) / (crab_par['alpha'] + 1)

    n_sim_bin = y

    # Rates and weights
    rate_g = rate("PowerLaw", mc_par_g['emin'], mc_par_g['emax'], crab_par, mc_par_g['cone'], mc_par_g['area_sim'])

    rate_p = rate("PowerLaw", mc_par_p['emin'], mc_par_p['emax'], proton_par, mc_par_p['cone'], mc_par_p['area_sim'])

    w_g = weight("PowerLaw", mc_par_g['emin'], mc_par_g['emax'], mc_par_g['sp_idx'], rate_g,
                 mc_par_g['sim_ev'], crab_par)

    w_p = weight("PowerLaw", mc_par_p['emin'], mc_par_p['emax'], mc_par_p['sp_idx'], rate_p,
                 mc_par_p['sim_ev'], proton_par)

    if (w_g.unit ==  u.Unit("sr / s")):
        print("You are using diffuse gammas to estimate point-like sensitivity")
        print("These results will make no sense")
        w_g = w_g / u.sr  # Fix to make tests pass

    rate_weighted_g = ((e_true_g / crab_par['e0']) ** (crab_par['alpha'] - mc_par_g['sp_idx'])) \
                      * w_g
    rate_weighted_p = ((e_true_p / proton_par['e0']) ** (proton_par['alpha'] - mc_par_p['sp_idx'])) \
                      * w_p
                      
                      
    p_contained, ang_area_p = ring_containment(angdist2_p, 0.4 * u.deg, 0.3 * u.deg)

    

    # FIX: ring_radius and ring_halfwidth should have units of deg
    # FIX: hardcoded at the moment, but ring_radius should be read from
    # the gamma file (point-like) or given as input (diffuse).
    # FIX: ring_halfwidth should be given as input
    area_ratio_p = np.pi * theta2_bins / ang_area_p
    # ratio between the area where we search for protons ang_area_p
    # and the area where we search for gammas math.pi * t

    # Arrays to contain the number of gammas and hadrons for different cuts
    final_gamma = np.ndarray(shape=(n_bins_energy, n_bins_gammaness, n_bins_theta2))
    final_hadrons = np.ndarray(shape=(n_bins_energy, n_bins_gammaness, n_bins_theta2))
    pre_gamma = np.ndarray(shape=(n_bins_energy, n_bins_gammaness, n_bins_theta2))
    pre_hadrons = np.ndarray(shape=(n_bins_energy, n_bins_gammaness, n_bins_theta2))

    ngamma_per_ebin = np.ndarray(n_bins_energy)
    nhadron_per_ebin = np.ndarray(n_bins_energy)

    total_rate_proton = np.sum(rate_weighted_p)
    total_rate_gamma = np.sum(rate_weighted_g)
    print("Total rate triggered proton {:.3f} Hz".format(total_rate_proton))
    print("Total rate triggered gamma  {:.3f} Hz".format(total_rate_gamma))

    # Weight events and count number of events per bin:
    for i in range(0, n_bins_energy):  # binning in energy
        total_rate_proton_ebin = np.sum(rate_weighted_p[(e_reco_p < energy[i + 1]) & (e_reco_p > energy[i])])

        print("\n******** Energy bin: {:.3f} - {:.3f} TeV ********".format(energy[i].value, energy[i + 1].value))
        total_rate_proton_ebin = np.sum(rate_weighted_p[(e_reco_p < energy[i+1]) & (e_reco_p > energy[i])])
        total_rate_gamma_ebin = np.sum(rate_weighted_g[(e_reco_g < energy[i+1]) & (e_reco_g > energy[i])])

        #print("**************")
        print("Total rate triggered proton in this bin {:.5f} Hz".format(total_rate_proton_ebin.value))
        print("Total rate triggered gamma in this bin {:.5f} Hz".format(total_rate_gamma_ebin.value))

        for j in range(0, n_bins_gammaness):  #  cut in gammaness
            for k in range(0, n_bins_theta2):  #  cut in theta2                
                rate_g_ebin = np.sum(rate_weighted_g[(e_reco_g < energy[i+1]) & (e_reco_g > energy[i]) \
                                            & (gammaness_g > gammaness_bins[j]) & (theta2_g < theta2_bins[k])])

                rate_p_ebin = np.sum(rate_weighted_p[(e_reco_p < energy[i+1]) & (e_reco_p > energy[i]) \
                                            & (gammaness_p > gammaness_bins[j]) & p_contained])
                final_gamma[i][j][k] = rate_g_ebin * obstime
                final_hadrons[i][j][k] = rate_p_ebin * obstime * area_ratio_p[k]

                pre_gamma[i][j][k] = e_reco_g[(e_reco_g < energy[i+1]) & (e_reco_g > energy[i]) \
                                            & (gammaness_g > gammaness_bins[j]) & (theta2_g < theta2_bins[k])].shape[0]
                pre_hadrons[i][j][k] = e_reco_p[(e_reco_p < energy[i+1]) & (e_reco_p > energy[i]) \
                                            & (gammaness_p > gammaness_bins[j]) & p_contained].shape[0]

                ngamma_per_ebin[i] = np.sum(rate_weighted_g[(e_reco_g < energy[i+1]) & (e_reco_g > energy[i])]) * obstime
                nhadron_per_ebin[i] = np.sum(rate_weighted_p[(e_reco_p < energy[i+1]) & (e_reco_p > energy[i])]) * obstime


    n_excesses_5sigma, sensitivity_3Darray = calculate_sensitivity_lima(final_gamma, final_hadrons * noff, 1/noff * np.ones_like(final_gamma), n_bins_energy, n_bins_gammaness, n_bins_theta2)
    
    # Avoid bins which are empty or have too few events:
    min_num_events = 5
    min_pre_events = 5

    # Minimum number of gamma and proton events in a bin to be taken into account for minimization
    for i in range(0, n_bins_energy):
        for j in range(0, n_bins_gammaness):
            for k in range(0, n_bins_theta2):
                conditions = (not np.isfinite(sensitivity_3Darray[i,j,k])) or (sensitivity_3Darray[i,j,k]<=0) \
                             or (final_hadrons[i,j,k] < min_num_events) \
                             or (pre_gamma[i,j,k] < min_pre_events) \
                             or (pre_hadrons[i,j,k] < min_pre_events)
                if conditions:
                    sensitivity_3Darray[i][j][k] = np.inf

    # Quantities to show in the results
    sensitivity = np.ndarray(shape = n_bins_energy)
    n_excesses_min = np.ndarray(shape = n_bins_energy)
    eff_g = np.ndarray(shape = n_bins_energy)
    eff_p = np.ndarray(shape = n_bins_energy)
    gcut = np.ndarray(shape = n_bins_energy)
    tcut = np.ndarray(shape = n_bins_energy)
    ngammas = np.ndarray(shape = n_bins_energy)
    nhadrons = np.ndarray(shape = n_bins_energy)
    gammarate = np.ndarray(shape = n_bins_energy)
    hadronrate = np.ndarray(shape = n_bins_energy)
    eff_area = np.ndarray(shape = n_bins_energy)
    nevents_gamma = np.ndarray(shape = n_bins_energy)
    nevents_proton = np.ndarray(shape = n_bins_energy)

    # Calculate the minimum sensitivity per energy bin
    for i in range(0, n_bins_energy):
        ind = np.unravel_index(np.nanargmin(sensitivity_3Darray[i], axis=None), sensitivity_3Darray[i].shape)
        gcut[i] = gammaness_bins[ind[0]]
        tcut[i] = theta2_bins[ind[1]].to_value()
        ngammas[i] = final_gamma[i][ind]
        nhadrons[i] = final_hadrons[i][ind]
        gammarate[i] = final_gamma[i][ind] / (obstime.to(u.min)).to_value()
        hadronrate[i] = final_hadrons[i][ind] / (obstime.to(u.min)).to_value()
        n_excesses_min[i] = n_excesses_5sigma[i][ind]
        sensitivity[i] = sensitivity_3Darray[i][ind]
        eff_g[i] = final_gamma[i][ind] / ngamma_per_ebin[i]
        eff_p[i] = final_hadrons[i][ind] / nhadron_per_ebin[i]

        e_aftercuts = e_true_g[(e_reco_g < energy[i + 1]) & (e_reco_g > energy[i]) \
                               & (gammaness_g > gammaness_bins[ind[0]]) & (theta2_g < theta2_bins[ind[1]])]


        e_aftercuts_p = e_true_p[(e_reco_p < energy[i + 1]) & (e_reco_p > energy[i]) \
                                 & p_contained]
        e_aftercuts_w = np.sum(np.power(e_aftercuts, crab_par['alpha'] - mc_par_g['sp_idx']))

        eff_area[i] = e_aftercuts_w.to_value() / n_sim_bin[i] * mc_par_g['area_sim'].to(u.m**2).to_value()

        nevents_gamma[i] = e_aftercuts.shape[0]
        nevents_proton[i] = e_aftercuts_p.shape[0]

    # Compute sensitivity in flux units
    egeom = np.sqrt(energy[1:] * energy[:-1])
    dFdE, par = crab_hegra(egeom)
    sensitivity_flux = sensitivity / 100 * (dFdE * egeom * egeom).to(u.erg / (u.cm**2 * u.s))

    list_of_tuples = list(zip(energy[:energy.shape[0]-2].to_value(), energy[1:].to_value(), gcut, tcut,
                            ngammas, nhadrons,
                            gammarate, hadronrate,
                            n_excesses_min, sensitivity_flux.to_value(), eff_area,
                              eff_g, eff_p, nevents_gamma, nevents_proton))
    result = pd.DataFrame(list_of_tuples,
                           columns=['ebin_low', 'ebin_up', 'gammaness_cut', 'theta2_cut',
                                    'n_gammas', 'n_hadrons',
                                    'gamma_rate', 'hadron_rate',
                                    'n_excesses_min', 'sensitivity','eff_area',
                                    'eff_gamma', 'eff_hadron',
                                    'nevents_g', 'nevents_p'])

    units = [energy.unit, energy.unit,"", theta2_bins.unit,"", "",
             u.min**-1, u.min**-1, "",
             sensitivity_flux.unit, mc_par_g['area_sim'].to(u.cm**2).unit, "", "", "", ""]
    
    # sensitivity_minimization_plot(n_bins_energy, n_bins_gammaness, n_bins_theta2, energy, sensitivity_3Darray)
    # plot_positions_survived_events(events_g,
    #                               events_p,
    #                               gammaness_g, gammaness_p,
    #                               theta2_g, p_contained, sensitivity_3Darray, energy, n_bins_energy, g, t)

    return energy, sensitivity, result, units, gcut, tcut


def sensitivity(simtelfile_gammas, simtelfile_protons,
                dl2_file_g, dl2_file_p,
                nfiles_gammas, nfiles_protons,
                n_bins_energy, gcut, tcut, noff,
                obstime=50 * 3600 * u.s):
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
    n_bins_energy: `int` number of bins in energy
    n_bins_gammaness: `int` number of bins in gammaness
    n_bins_theta2: `int` number of bins in theta2
    noff: `float` ratio between the background and the signal region
    obstime: `Quantity` Observation time in seconds

    Returns
    ---------
    energy: `array` center of energy bins
    sensitivity: `array` sensitivity per energy bin

    """

    # Read simulated and reconstructed values
    gammaness_g, theta2_g, e_reco_g, e_true_g, mc_par_g, events_g = process_mc(simtelfile_gammas,
                                                                               dl2_file_g, 'gamma')
    gammaness_p, angdist2_p, e_reco_p, e_true_p, mc_par_p, events_p = process_mc(simtelfile_protons,
                                                                                 dl2_file_p, 'proton')

    mc_par_g['sim_ev'] = mc_par_g['sim_ev'] * nfiles_gammas
    mc_par_p['sim_ev'] = mc_par_p['sim_ev'] * nfiles_protons

    # Pass units to TeV and cm2
    mc_par_g['emin'] = mc_par_g['emin'].to(u.TeV)
    mc_par_g['emax'] = mc_par_g['emax'].to(u.TeV)

    mc_par_p['emin'] = mc_par_p['emin'].to(u.TeV)
    mc_par_p['emax'] = mc_par_p['emax'].to(u.TeV)

    mc_par_g['area_sim'] = mc_par_g['area_sim'].to(u.cm ** 2)
    mc_par_p['area_sim'] = mc_par_p['area_sim'].to(u.cm ** 2)

    # Set binning for sensitivity calculation
    emin_sensitivity = 0.01 * u.TeV  # mc_par_g['emin'] 
    emax_sensitivity =  100 * u.TeV  # mc_par_g['emax']

    energy = np.logspace(np.log10(emin_sensitivity.to_value()),
                         np.log10(emax_sensitivity.to_value()), n_bins_energy + 1) * u.TeV

    # Extract spectral parameters
    dFdE, crab_par = crab_hegra(energy)
    dFdEd0, proton_par = proton_bess(energy)

    bins = np.logspace(np.log10(emin_sensitivity.to_value()), np.log10(emax_sensitivity.to_value()), n_bins_energy + 1)
    y0 = mc_par_g['sim_ev'] / (mc_par_g['emax'].to_value() ** (mc_par_g['sp_idx'] + 1) \
                               - mc_par_g['emin'].to_value() ** (mc_par_g['sp_idx'] + 1)) \
         * (mc_par_g['sp_idx'] + 1)
    y = y0 * (bins[1:] ** (crab_par['alpha'] + 1) - bins[:-1] ** (crab_par['alpha'] + 1)) / (crab_par['alpha'] + 1)

    n_sim_bin = y

    # Rates and weights
    rate_g = rate("PowerLaw", mc_par_g['emin'], mc_par_g['emax'], crab_par, mc_par_g['cone'], mc_par_g['area_sim'])

    rate_p = rate("PowerLaw", mc_par_p['emin'], mc_par_p['emax'], proton_par, mc_par_p['cone'], mc_par_p['area_sim'])

    w_g = weight("PowerLaw", mc_par_g['emin'], mc_par_g['emax'], mc_par_g['sp_idx'], rate_g,
                 mc_par_g['sim_ev'], crab_par)

    w_p = weight("PowerLaw", mc_par_p['emin'], mc_par_p['emax'], mc_par_p['sp_idx'], rate_p,
                 mc_par_p['sim_ev'], proton_par)
    
    if (w_g.unit ==  u.Unit("sr / s")):
        print("You are using diffuse gammas to estimate point-like sensitivity")
        print("These results will make no sense")
        w_g = w_g / u.sr  # Fix to make tests pass

    rate_weighted_g = ((e_true_g / crab_par['e0']) ** (crab_par['alpha'] - mc_par_g['sp_idx'])) \
                      * w_g
    rate_weighted_p = ((e_true_p / proton_par['e0']) ** (proton_par['alpha'] - mc_par_p['sp_idx'])) \
                      * w_p

    p_contained, ang_area_p = ring_containment(angdist2_p, 0.4 * u.deg, 0.3 * u.deg)

    # FIX: ring_radius and ring_halfwidth should have units of deg
    # FIX: hardcoded at the moment, but ring_radius should be read from
    # the gamma file (point-like) or given as input (diffuse).
    # FIX: ring_halfwidth should be given as input
    area_ratio_p = np.pi * tcut / ang_area_p
    # ratio between the area where we search for protons ang_area_p
    # and the area where we search for gammas math.pi * t

    # Arrays to contain the number of gammas and hadrons for different cuts
    final_gamma = np.ndarray(shape=(n_bins_energy))
    final_hadrons = np.ndarray(shape=(n_bins_energy))
    pre_gamma = np.ndarray(shape=(n_bins_energy))
    pre_hadrons = np.ndarray(shape=(n_bins_energy))

    ngamma_per_ebin = np.ndarray(n_bins_energy)
    nhadron_per_ebin = np.ndarray(n_bins_energy)

    # Weight events and count number of events per bin:
    for i in range(n_bins_energy):  # binning in energy
        rate_g_ebin = np.sum(rate_weighted_g[(e_reco_g < energy[i + 1]) & (e_reco_g > energy[i]) \
                                             & (gammaness_g > gcut[i]) & (theta2_g < tcut[i])])


        rate_p_ebin = np.sum(rate_weighted_p[(e_reco_p < energy[i + 1]) & (e_reco_p > energy[i]) \
                                             & (gammaness_p > gcut[i]) & p_contained])
        final_gamma[i] = (rate_g_ebin * obstime).value
        final_hadrons[i] = (rate_p_ebin * obstime).value * area_ratio_p[i]

        pre_gamma[i] = e_reco_g[(e_reco_g < energy[i + 1]) & (e_reco_g > energy[i]) \
                                & (gammaness_g > gcut[i]) & (theta2_g < tcut[i])].shape[0]
        pre_hadrons[i] = e_reco_p[(e_reco_p < energy[i + 1]) & (e_reco_p > energy[i]) \
                                  & (gammaness_p > gcut[i]) & p_contained].shape[0]

        ngamma_per_ebin[i] = np.sum(
            rate_weighted_g[(e_reco_g < energy[i + 1]) & (e_reco_g > energy[i])].to(1 / u.s).value) \
                             * obstime.to(u.s).value
        nhadron_per_ebin[i] = np.sum(
            rate_weighted_p[(e_reco_p < energy[i + 1]) & (e_reco_p > energy[i])].to(1 / u.s).value) \
                              * obstime.to(u.s).value

    n_excesses_5sigma, sensitivity_3Darray = calculate_sensitivity_lima_ebin(final_gamma, final_hadrons * noff,
                                                                             1 / noff * np.ones(len(final_gamma)),
                                                                             n_bins_energy)
    # Avoid bins which are empty or have too few events:
    min_num_events = 5
    min_pre_events = 5
    # Minimum number of gamma and proton events in a bin to be taken into account for minimization
    for i in range(0, n_bins_energy):
        conditions = (not np.isfinite(sensitivity_3Darray[i])) or (sensitivity_3Darray[i] <= 0) \
                     or (final_hadrons[i] < min_num_events) \
                     or (pre_gamma[i] < min_pre_events) \
                     or (pre_hadrons[i] < min_pre_events)
        if conditions:
            sensitivity_3Darray[i] = np.inf

    # Quantities to show in the results
    sensitivity = np.ndarray(shape=n_bins_energy)
    n_excesses_min = np.ndarray(shape=n_bins_energy)
    eff_g = np.ndarray(shape=n_bins_energy)
    eff_p = np.ndarray(shape=n_bins_energy)
    ngammas = np.ndarray(shape=n_bins_energy)
    nhadrons = np.ndarray(shape=n_bins_energy)
    gammarate = np.ndarray(shape=n_bins_energy)
    hadronrate = np.ndarray(shape=n_bins_energy)
    eff_area = np.ndarray(shape=n_bins_energy)
    nevents_gamma = np.ndarray(shape=n_bins_energy)
    nevents_proton = np.ndarray(shape=n_bins_energy)

    # Calculate the minimum sensitivity per energy bin
    for i in range(0, n_bins_energy):
        ngammas[i] = final_gamma[i]
        nhadrons[i] = final_hadrons[i]
        gammarate[i] = final_gamma[i] / (obstime.to(u.min)).to_value()
        hadronrate[i] = final_hadrons[i] / (obstime.to(u.min)).to_value()
        n_excesses_min[i] = n_excesses_5sigma[i]
        sensitivity[i] = sensitivity_3Darray[i]
        eff_g[i] = final_gamma[i] / ngamma_per_ebin[i]
        eff_p[i] = final_hadrons[i] / nhadron_per_ebin[i]

        e_aftercuts = e_true_g[(e_reco_g < energy[i + 1]) & (e_reco_g > energy[i]) \
                               & (gammaness_g > gcut[i]) & (theta2_g < tcut[i])]

        e_aftercuts_p = e_true_p[(e_reco_p < energy[i + 1]) & (e_reco_p > energy[i]) \
                                 & (gammaness_p > gcut[i]) & p_contained]

        e_aftercuts_w = np.sum(np.power(e_aftercuts, crab_par['alpha'] - mc_par_g['sp_idx']))

        e_w = np.sum(np.power(e_true_g[(e_reco_g < energy[i + 1]) & (e_reco_g > energy[i])],
                              crab_par['alpha'] - mc_par_g['sp_idx']))

        eff_area[i] = e_aftercuts_w.to_value() / n_sim_bin[i] * mc_par_g['area_sim'].to(u.m ** 2).to_value()

        nevents_gamma[i] = e_aftercuts.shape[0]
        nevents_proton[i] = e_aftercuts_p.shape[0]

    # Compute sensitivity  in flux units

    egeom = np.sqrt(energy[1:] * energy[:-1])
    dFdE, par = crab_hegra(egeom)
    sensitivity_flux = sensitivity / 100 * (dFdE * egeom * egeom).to(u.erg / (u.cm ** 2 * u.s))
    
    print("\n******** Energy [TeV] *********\n")
    print(egeom)
    print("\nsensitivity flux:\n", sensitivity_flux)
    print("\nsensitivity[%]:\n", sensitivity)
    print("\n**************\n")
    
    list_of_tuples = list(zip(energy[:energy.shape[0] - 2].to_value(), energy[1:].to_value(), gcut, tcut,
                              ngammas, nhadrons,
                              gammarate, hadronrate,
                              n_excesses_min, sensitivity_flux.to_value(), eff_area,
                              eff_g, eff_p, nevents_gamma, nevents_proton))
    result = pd.DataFrame(list_of_tuples,
                          columns=['ebin_low', 'ebin_up', 'gammaness_cut', 'theta2_cut',
                                   'n_gammas', 'n_hadrons',
                                   'gamma_rate', 'hadron_rate',
                                   'n_excesses_min', 'sensitivity', 'eff_area',
                                   'eff_gamma', 'eff_hadron',
                                   'nevents_g', 'nevents_p'])

    units = [energy.unit, energy.unit, "", tcut.unit, "", "",
             u.min ** -1, u.min ** -1, "",
             sensitivity_flux.unit, mc_par_g['area_sim'].to(u.cm ** 2).unit, "", "", "", ""]

    # sensitivity_minimization_plot(n_bins_energy, n_bins_gammaness, n_bins_theta2, energy, sensitivity)
    # plot_positions_survived_events(events_g,
    #                                events_p,
    #                                gammaness_g, gammaness_p,
    #                                theta2_g, p_contained, sensitivity, energy, n_bins_energy, gcut, tcut)

    # Build dataframe of events that survive the cuts:
    events = pd.concat((events_g, events_p))
    dl2 = pd.DataFrame(columns=events.keys())
    
    for i in range(0, n_bins_energy):
        df_bin = events[(events.mc_energy < energy[i+1]) & (events.mc_energy > energy[i]) \
                               & (events.gammaness > gcut[i]) & (events.theta2 < tcut[i])]

        dl2 = pd.concat((dl2, df_bin))

    return energy, sensitivity, result, units, dl2
