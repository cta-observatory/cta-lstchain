import numpy as np
import pandas as pd
import astropy.units as u
from .mc import rate, weight
from lstchain.spectra.crab import crab_hegra
from lstchain.spectra.proton import proton_bess
from lstchain.reco.utils import reco_source_position_sky, get_effective_time
from astropy.coordinates.angle_utilities import angular_separation
from lstchain.io import read_simu_info_merged_hdf5
from lstchain.io.io import dl2_params_lstcam_key
from pyirf.sensitivity import relative_sensitivity
from gammapy.stats import WStatCountsStatistic

__all__ = [
    'read_sim_par',
    'process_mc',
    'process_real',
    'get_weights',
    'samesign',
    'diff_events_after_cut',
    'find_cut',
    'calculate_sensitivity',
    'calculate_sensitivity_lima',
    'calculate_sensitivity_lima_ebin',
    'bin_definition',
    'ring_containment',
    'sensitivity_gamma_efficiency',
    'sensitivity_gamma_efficiency_real_protons',
    'sensitivity_gamma_efficiency_real_data',
    ]

def read_sim_par(file):

    """
    Read MC simulated parameters

    Parameters
    ---------
    file: `hdf5 file`

    Returns
    ---------
    par: `dict` with simulated parameters

    """
    simu_info = read_simu_info_merged_hdf5(file)
    emin = simu_info.energy_range_min
    emax = simu_info.energy_range_max
    sp_idx = simu_info.spectral_index
    sim_ev = simu_info.num_showers * simu_info.shower_reuse
    area_sim = (simu_info.max_scatter_range - simu_info.min_scatter_range) ** 2 * np.pi
    cone = simu_info.max_viewcone_radius

    par_var = [emin, emax, sp_idx, sim_ev, area_sim, cone]
    par_dic = ['emin', 'emax', 'sp_idx', 'sim_ev', 'area_sim', 'cone']
    par = dict(zip(par_dic, par_var))

    # Pass units to TeV and cm2
    par['emin'] = par['emin'].to(u.TeV)
    par['emax'] = par['emax'].to(u.TeV)
    par['area_sim'] = par['area_sim'].to(u.cm ** 2)

    return par


def process_mc(dl2_file, mc_type):
    """
    Process the MC simulated and reconstructed to extract the relevant
    parameters to compute the sensitivity

    Paramenters
    ---------
    dl2_file:  dl2 file with mc parameters
    events: `pandas DataFrame' dl2 events
    mc_type: 'string' type of particle

    Returns
    ---------
    gammaness: `numpy.ndarray`
    angdist2:  `numpy.ndarray` angular distance squared
    e_reco:    `numpy.ndarray` reconstructed energies
    n_reco:    `int` number of reconstructed events
    mc_par:    `dict` with simulated parameters

    """
    sim_par = read_sim_par(dl2_file)

    events = pd.read_hdf(dl2_file, key = dl2_params_lstcam_key)

    # Filters:
    # TO DO: These cuts must be given in a configuration file
    # By now: only cut in leakage and intensity
    # we use all telescopes (number of events needs to be multiplied
    # by the number of LSTs in the simulation)

    filter_good_events = (
        (events.leakage_intensity_width_2 < 0.2)
        & (events.intensity > 100)
    )
    events = events[filter_good_events]

    e_reco = events.reco_energy.to_numpy() * u.TeV
    e_true = events.mc_energy.to_numpy() * u.TeV

    gammaness = events.gammaness

    # If the particle is a gamma ray, it returns the squared angular distance
    # from the reconstructed gamma-ray position and the simulated incoming position
    if mc_type == 'gamma':
        alt2 = events.mc_alt
        az2 = events.mc_az

    # If the particle is not a gamma-ray (diffuse protons/electrons), it returns
    # the squared angular distance of the reconstructed position w.r.t. the
    # center of the camera
    else:
        alt2 = events.mc_alt_tel
        az2 = events.mc_az_tel

    alt1=events.reco_alt
    az1=events.reco_az

    angdist2 = (angular_separation(az1, alt1, az2, alt2).to_numpy() * u.rad) ** 2
    events['theta2'] = angdist2.to(u.deg**2)

    return gammaness, angdist2.to(u.deg**2), e_reco, e_true, sim_par, events

def process_real(dl2_file):

    events = pd.read_hdf(dl2_file, key = dl2_params_lstcam_key)
    obstime_real = get_effective_time(events)[0]

    filter_good_events = (
        (events.leakage_intensity_width_2 < 0.2)
        & (events.intensity > 100)
    )
    events = events[filter_good_events]

    e_reco = events.reco_energy.to_numpy() * u.TeV
    gammaness = events.gammaness
    # If the particle is a gamma ray, it returns the squared angular distance
    # from the reconstructed gamma-ray position and the simulated incoming position
    alt2 = events.alt_tel
    az2 = events.az_tel

    alt1=events.reco_alt
    az1=events.reco_az

    angdist2 = (angular_separation(az1, alt1, az2, alt2).to_numpy() * u.rad) ** 2
    events['theta2'] = angdist2.to(u.deg**2)

    return gammaness, angdist2.to(u.deg**2),e_reco, events, obstime_real

def get_weights(mc_par, spectral_par):
    """
    Calculate the weight to transform from MC spectra to target spectra

    Paramenters
    ---------
    mc_par:  `dict` MC spectral parameters
    spectral_par: `dict`spectral parameters of desired spectrum

    Returns
    ---------
    w: `float` weight

    """
    r = rate("PowerLaw",
                  mc_par['emin'], mc_par['emax'],
                  spectral_par, mc_par['cone'], mc_par['area_sim'])

    w = weight("PowerLaw",
                 mc_par['emin'], mc_par['emax'],
                 mc_par['sp_idx'], r,
                 mc_par['sim_ev'], spectral_par)
    return w

def diff_events_after_cut_real(events_on, events_off, obstime_on, obstime_off, feature, cut, gamma_efficiency):

    total_signal=events_on.shape[0] - (events_off.shape[0]*obstime_on/obstime_off)
    if feature=="gammaness":
        events_on_after_cut=events_on[events_on[feature]>cut].shape[0]
        events_off_after_cut=events_off[events_off[feature]>cut].shape[0]*obstime_on/obstime_off

    else:
        events_on_after_cut=events_on[events_on[feature]<cut].shape[0]
        events_off_after_cut=events_off[events_off[feature]<cut].shape[0]*obstime_on/obstime_off

    signal_after_cut=events_on_after_cut-events_off_after_cut

    print(signal_after_cut, total_signal, gamma_efficiency*total_signal)
    return gamma_efficiency*total_signal-signal_after_cut


def diff_events_after_cut(events, rates, obstime, feature, cut, gamma_efficiency):
    """
    This function calculates the difference between the number of events after the cut
    in feature and gamma_efficiency*total number of events

    Paramenters
    ---------
    events:  `pd.dataframe` Dataframe of events
    rates: `np.ndarray` gamma rates
    obstime: `observation time`
    feature: `string` feature for cut: gammaness or theta2
    cut: `float` cut in feature
    gamma_efficiency: `float` target gamma efficiency for the cut

    Returns
    ---------
    midpoint: `float` cut in feature

    """

    total_events=np.sum(rates) * obstime


    if feature=="gammaness":
        events_after_cut=np.sum(rates[events[feature]>cut]) * obstime

    else:
        events_after_cut=np.sum(rates[events[feature]<cut]) * obstime

    return gamma_efficiency*total_events-events_after_cut


def samesign(a,b):
    """
    Check if two numbers have the same sign
    Paramenters
    ---------
    a: `float`
    b: `float`

    Returns
    ---------
    a * b > 0: `bool` True if a and b have the same sign

    """
    return a * b > 0

def find_cut(events, rates, obstime, feature, low_cut, high_cut, gamma_efficiency):
    """
    Find cut in feature that corresponds to gamma efficiency.
    Bisection method is used to find the root of the function
    Number of events after cuts - gamma_efficiency*total number of events

    Paramenters
    ---------
    events:  `pd.dataframe` Dataframe of events
    rates: `np.ndarray` gamma rates
    obstime: `observation time`
    feature: `string` feature for cut: gammaness or theta2
    low_cut: `float` lower cut limit
    high_cut: `float` higher cut limit
    gamma_efficiency: `float` target gamma efficiency for the cut

    Returns
    ---------
    midpoint: `float` cut in feature

    """

    if events.shape[0] == 0:

        if feature=="gammaness":
            return low_cut
        else:
            return high_cut


    tol = 1000

    if feature=="gammaness":
        lookfor_cut = high_cut
        alternative_cut = low_cut
    else:
        lookfor_cut = low_cut
        alternative_cut = high_cut

    while tol > 1e-6:
        midpoint = (lookfor_cut + alternative_cut) / 2.0

        if samesign(diff_events_after_cut(events, rates, obstime, feature, lookfor_cut, gamma_efficiency),
                    diff_events_after_cut(events, rates, obstime, feature, midpoint, gamma_efficiency)):
            lookfor_cut = midpoint
        else:
            alternative_cut = midpoint

        tol = abs(alternative_cut -lookfor_cut)
    return midpoint

def find_cut_real(events_on, events_off, obstime_on, obstime_off, feature, low_cut, high_cut, gamma_efficiency):
    """
    Find cut in feature that corresponds to gamma efficiency.
    Bisection method is used to find the root of the function
    Number of events after cuts - gamma_efficiency*total number of events

    Paramenters
    ---------
    events:  `pd.dataframe` Dataframe of events
    rates: `np.ndarray` gamma rates
    obstime: `observation time`
    feature: `string` feature for cut: gammaness or theta2
    low_cut: `float` lower cut limit
    high_cut: `float` higher cut limit
    gamma_efficiency: `float` target gamma efficiency for the cut

    Returns
    ---------
    midpoint: `float` cut in feature

    """
    if events_on.shape[0] == 0:
        if feature=="gammaness":
            return low_cut
        else:
            return high_cut

    tol = 1000

    if feature=="gammaness":
        lookfor_cut = high_cut
        alternative_cut = low_cut
    else:
        lookfor_cut = low_cut
        alternative_cut = high_cut

    while tol > 1e-6:
        midpoint = (lookfor_cut + alternative_cut) / 2.0

        if samesign(diff_events_after_cut_real(events_on, events_off, obstime_on, obstime_off, feature, lookfor_cut, gamma_efficiency),
                    diff_events_after_cut_real(events_on, events_off, obstime_on, obstime_off, feature, midpoint, gamma_efficiency)):
            lookfor_cut = midpoint
        else:
            alternative_cut = midpoint

        tol = abs(alternative_cut -lookfor_cut)

    return midpoint


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

def calculate_sensitivity_lima(n_signal, n_background, alpha):
    """
    Sensitivity calculation using the Li & Ma formula
    eq. 17 of Li & Ma (1983).
    https://ui.adsabs.harvard.edu/abs/1983ApJ...272..317L/abstract

    We calculate the sensitivity in bins of energy and
    theta2

    Parameters
    ---------
    n_on_events:   `numpy.ndarray` number of ON events in the signal region
    n_background:   `numpy.ndarray` number of events in the background region
    alpha: `float` inverse of the number of off positions
    n_bins_energy: `int` number of bins in energy
    n_bins_theta2: `int` number of bins in theta2

    Returns
    ---------
    sensitivity: `numpy.ndarray` sensitivity in percentage of Crab units
    n_excesses_5sigma: `numpy.ndarray` number of excesses corresponding to
                a 5 sigma significance

    """

    stat = WStatCountsStatistic(
        n_on=n_signal+alpha*n_background,
        n_off=n_background,
        alpha=alpha
        )
    n_excesses_5sigma = stat.n_sig_matching_significance(5)
    n_excesses_5sigma[n_excesses_5sigma<10] = 10
    bkg_5percent = 0.05*n_background*alpha
    n_excesses_5sigma[n_excesses_5sigma<bkg_5percent] = bkg_5percent[n_excesses_5sigma<bkg_5percent]

    sensitivity = n_excesses_5sigma / (n_signal) * 100  # percentage of Crab

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

    stat = WStatCountsStatistic(
        n_on=n_on_events,
        n_off=n_background,
        alpha=alpha
        )

    n_excesses_5sigma = stat.n_sig_matching_significance(5)

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
    max_gam = 0.9
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

def sensitivity_gamma_efficiency(dl2_file_g, dl2_file_p,
                ntelescopes_gammas, ntelescopes_protons,
                n_bins_energy,
                gamma_eff_gammaness,
                gamma_eff_theta2,
                noff,
                obstime = 50 * 3600 * u.s):

    """
    Main function to calculate the sensitivity for cuts based
    on gamma efficiency

    Parameters
    ---------
    dl2_file_g: `string` path to h5 file of reconstructed gammas
    dl2_file_p: `string' path to h5 file of reconstructed protons
    ntelescopes_gammas: `int` number of telescopes used
    ntelescopes_protons: `int` number of telescopes used
    n_bins_energy: `int` number of bins in energy
    gamma_eff_gammaness: `float` between 0 and 1 %/100
    of gammas to be left after cut in gammaness
    gamma_eff_theta2: `float` between 0 and 1 %/100
    of gammas to be left after cut in theta2
    noff: `float` ratio between the background and the signal region
    obstime: `Quantity` Observation time in seconds

    Returns
    ---------
    energy: `array` center of energy bins
    sensitivity: `array` sensitivity per energy bin

    """

    # Read simulated and reconstructed values

    gammaness_g, theta2_g, e_reco_g, e_true_g, mc_par_g, events_g = process_mc(dl2_file_g, 'gamma')
    gammaness_p, angdist2_p, e_reco_p, e_true_p, mc_par_p, events_p = process_mc(dl2_file_p, 'proton')

    #Account for the number of telescopes simulated
    mc_par_g['sim_ev'] = mc_par_g['sim_ev'] * ntelescopes_gammas
    mc_par_p['sim_ev'] = mc_par_p['sim_ev'] * ntelescopes_protons

    # Set binning for sensitivity calculation
    emin_sensitivity =  mc_par_p['emin']
    emax_sensitivity =  mc_par_p['emax']

    #Energy bins
    energy = np.logspace(np.log10(emin_sensitivity.to_value()),
                         np.log10(emax_sensitivity.to_value()), n_bins_energy + 1) * u.TeV

    # Extract spectral parameters
    dFdE, crab_par = crab_hegra(energy)
    dFdEd0, proton_par = proton_bess(energy)

    # Rates and weights

    w_g = get_weights(mc_par_g, crab_par)
    w_p = get_weights(mc_par_p, proton_par)

    if (w_g.unit ==  u.Unit("sr / s")):
        print("You are using diffuse gammas to estimate point-like sensitivity")
        print("These results will make no sense")
        w_g = w_g / u.sr  # Fix to make tests pass

    rate_weighted_g = ((e_true_g / crab_par['e0']) ** (crab_par['alpha'] - mc_par_g['sp_idx'])) \
                      * w_g
    rate_weighted_p = ((e_true_p / proton_par['e0']) ** (proton_par['alpha'] - mc_par_p['sp_idx'])) \
                      * w_p

    #For background, select protons contained in a ring overlapping with the ON region
    p_contained, ang_area_p = ring_containment(angdist2_p, 1.0 * u.deg, 0.9 * u.deg)
    # FIX: ring_radius and ring_halfwidth should have units of deg
    # FIX: hardcoded at the moment, but ring_radius should be read from
    # the gamma file (point-like) or given as input (diffuse).
    # FIX: ring_halfwidth should be given as input

    # Initialize arrays

    final_gammas = np.ndarray(shape=(n_bins_energy))
    final_protons = np.ndarray(shape=(n_bins_energy))
    pre_gammas = np.ndarray(shape=(n_bins_energy))
    pre_protons = np.ndarray(shape=(n_bins_energy))
    weighted_gamma_per_ebin = np.ndarray(n_bins_energy)
    weighted_proton_per_ebin = np.ndarray(n_bins_energy)
    sensitivity = np.ndarray(shape = n_bins_energy)
    n_excesses_min = np.ndarray(shape = n_bins_energy)
    eff_g = np.ndarray(shape = n_bins_energy)
    eff_p = np.ndarray(shape = n_bins_energy)
    gcut = np.ndarray(shape = n_bins_energy)
    tcut = np.ndarray(shape = n_bins_energy)
    gamma_rate = np.ndarray(shape = n_bins_energy)
    proton_rate = np.ndarray(shape = n_bins_energy)

    #Total rate of gammas and protons
    total_rate_proton = np.sum(rate_weighted_p)
    total_rate_gamma = np.sum(rate_weighted_g)

    print("Total rate triggered proton {:.3f} Hz".format(total_rate_proton))
    print("Total rate triggered gamma  {:.3f} Hz".format(total_rate_gamma))

    #Dataframe to store the events which survive the cuts
    gammalike_events = pd.DataFrame(columns=events_g.keys())

    # Weight events and count number of events per bin:
    for i in range(0, n_bins_energy):  # binning in energy

        print("\n******** Energy bin: {:.3f} - {:.3f} TeV ********".format(energy[i].value, energy[i + 1].value))
        total_rate_proton_ebin = np.sum(rate_weighted_p[(e_reco_p < energy[i+1]) & (e_reco_p > energy[i])])
        total_rate_gamma_ebin = np.sum(rate_weighted_g[(e_reco_g < energy[i+1]) & (e_reco_g > energy[i])])

        #print("**************")
        print("Total rate triggered proton in this bin {:.5f} Hz".format(total_rate_proton_ebin.value))
        print("Total rate triggered gamma in this bin {:.5f} Hz".format(total_rate_gamma_ebin.value))

        #Calculate the cuts in gammaness and theta2 based on efficiency of weighted gammas

        rates_g = rate_weighted_g[(e_reco_g < energy[i+1]) & (e_reco_g > energy[i])]
        events_bin_g = events_g[(e_reco_g < energy[i+1]) & (e_reco_g > energy[i])]
        events_bin_p = events_p[(e_reco_p < energy[i+1]) & (e_reco_p > energy[i])]

        best_g_cut = find_cut(events_bin_g, rates_g, obstime,  "gammaness", 0.1, 1.0, gamma_eff_gammaness)
        best_theta2_cut = find_cut(events_bin_g, rates_g, obstime, "theta2", 0.0, 10.0, gamma_eff_theta2) * u.deg**2

        events_bin_after_cuts_g = events_bin_g[(events_bin_g.gammaness > best_g_cut) &(events_bin_g.theta2 < best_theta2_cut)]
        events_bin_after_cuts_p = events_bin_p[(events_bin_p.gammaness > best_g_cut) &(events_bin_p.theta2 < best_theta2_cut)]

        #Save the survived events in the dataframe
        gammalike_events = pd.concat((gammalike_events, events_bin_after_cuts_g))
        gammalike_events = pd.concat((gammalike_events, events_bin_after_cuts_p))


        # ratio between the area where we search for protons ang_area_p
        # and the area where we search for gammas math.pi * t
        area_ratio_p = np.pi * best_theta2_cut / ang_area_p

        rate_g_ebin = np.sum(rate_weighted_g[(e_reco_g < energy[i+1]) & (e_reco_g > energy[i]) \
                                             & (gammaness_g > best_g_cut) & (theta2_g < best_theta2_cut)])


        rate_p_ebin = np.sum(rate_weighted_p[(e_reco_p < energy[i+1]) & (e_reco_p > energy[i]) \
                                             & (gammaness_p > best_g_cut) & p_contained])

        gamma_rate[i] = rate_g_ebin.to(1/u.min).to_value()
        proton_rate[i] = rate_p_ebin.to(1/u.min).to_value()

        final_gammas[i] = rate_g_ebin * obstime
        final_protons[i] = rate_p_ebin * obstime * area_ratio_p

        pre_gammas[i] = e_reco_g[(e_reco_g < energy[i+1]) & (e_reco_g > energy[i]) \
                                   & (gammaness_g > best_g_cut) & (theta2_g < best_theta2_cut)].shape[0]
        pre_protons[i] = e_reco_p[(e_reco_p < energy[i+1]) & (e_reco_p > energy[i]) \
                                     & (gammaness_p > best_g_cut) & p_contained].shape[0]

        weighted_gamma_per_ebin[i] = np.sum(rate_weighted_g[(e_reco_g < energy[i+1]) & \
                                                    (e_reco_g > energy[i])]) * obstime
        weighted_proton_per_ebin[i] = np.sum(rate_weighted_p[(e_reco_p < energy[i+1]) & \
                                                     (e_reco_p > energy[i])]) * obstime

        gcut[i] = best_g_cut
        tcut[i] = best_theta2_cut.to_value()


        eff_g[i] = final_gammas[i] / weighted_gamma_per_ebin[i]
        eff_p[i] = final_protons[i] / weighted_proton_per_ebin[i]

    n_excesses_min, sensitivity = calculate_sensitivity_lima(final_gammas, final_protons*noff,
                                                             1/noff * np.ones_like(final_gammas))

    # Avoid bins which are empty or have too few events:
    min_num_events = 10
    min_pre_events = 5

    # Set conditions for calculating sensitivity

    conditions = (
         (sensitivity<=0)
        | (pre_gammas<min_pre_events)
        | (pre_protons<min_pre_events)
        | (final_gammas<min_num_events)
    )

    sensitivity[conditions] = np.inf

    # Compute sensitivity in flux units
    egeom = np.sqrt(energy[1:] * energy[:-1])
    dFdE, par = crab_hegra(egeom)
    sensitivity_flux = sensitivity / 100 * (dFdE * egeom * egeom).to(u.TeV / (u.cm**2 * u.s))

    print("\n******** Energy [TeV] *********\n")
    print(egeom)
    print("\nsensitivity flux:\n", sensitivity_flux)
    print("\nsensitivity[%]:\n", sensitivity)
    print("\n**************\n")

    list_of_tuples = list(zip(energy[:energy.shape[0]-1].to_value(), energy[1:].to_value(), gcut, tcut,
                            final_gammas, final_protons,
                            gamma_rate, proton_rate,
                              n_excesses_min, sensitivity, sensitivity_flux.to_value(),
                            eff_g, eff_p, pre_gammas, pre_protons))

    result = pd.DataFrame(list_of_tuples,
                           columns=['ebin_low', 'ebin_up', 'gammaness_cut', 'theta2_cut',
                                    'gammas_reweighted', 'protons_reweighted',
                                    'gamma_rate', 'proton_rate',
                                    'n_excesses_min', 'relative_sensitivity', 'sensitivity_flux',
                                    'eff_gamma', 'eff_proton',
                                    'mc_gammas', 'mc_protons'])

    return energy, sensitivity, result, gammalike_events, gcut, tcut

def sensitivity_gamma_efficiency_real_protons(dl2_file_g, dl2_file_p,
                ntelescopes_gammas,
                n_bins_energy,
                gamma_eff_gammaness,
                gamma_eff_theta2,
                noff,
                obstime = 50 * 3600 * u.s):

    """
    Main function to calculate the sensitivity for cuts based
    on gamma efficiency using real protons as background events

    Parameters
    ---------
    dl2_file_g: `string` path to h5 file of reconstructed gammas
    dl2_file_p: `string' path to h5 file of reconstructed real protons
    ntelescopes_gammas: `int` number of telescopes used
    ntelescopes_protons: `int` number of telescopes used
    n_bins_energy: `int` number of bins in energy
    gamma_eff_gammaness: `float` between 0 and 1 %/100
    of gammas to be left after cut in gammaness
    gamma_eff_theta2: `float` between 0 and 1 %/100
    of gammas to be left after cut in theta2
    noff: `float` ratio between the background and the signal region
    obstime: `Quantity` Observation time in seconds

    Returns
    ---------
    energy: `array` center of energy bins
    sensitivity: `array` sensitivity per energy bin

    """

    # Read simulated and reconstructed values

    gammaness_g, theta2_g, e_reco_g, e_true_g, mc_par_g, events_g = process_mc(dl2_file_g, 'gamma')
    gammaness_p, angdist2_p, e_reco_p, events_p, obstime_real = process_real(dl2_file_p)
    e_reco_p = events_p["reco_energy"]
    gammaness_p = events_p["gammaness"]

    #Account for the number of telescopes simulated
    mc_par_g['sim_ev'] = mc_par_g['sim_ev'] * ntelescopes_gammas

    # Set binning for sensitivity calculation
    emin_sensitivity =  mc_par_g['emin']
    emax_sensitivity =  mc_par_g['emax']

    #Energy bins
    energy = np.logspace(np.log10(emin_sensitivity.to_value()),
                         np.log10(emax_sensitivity.to_value()), n_bins_energy + 1) * u.TeV

    # Extract spectral parameters
    dFdE, crab_par = crab_hegra(energy)

    # Rates and weights
    w_g = get_weights(mc_par_g, crab_par)

    if (w_g.unit ==  u.Unit("sr / s")):
        print("You are using diffuse gammas to estimate point-like sensitivity")
        print("These results will make no sense")
        w_g = w_g / u.sr  # Fix to make tests pass

    rate_weighted_g = ((e_true_g / crab_par['e0']) ** (crab_par['alpha'] - mc_par_g['sp_idx'])) \
                      * w_g

    #For background, select protons contained in a ring overlapping with the ON region
    p_contained, ang_area_p = ring_containment(angdist2_p, 0.5 * u.deg, 0.5 * u.deg)
    #p_contained, ang_area_p = ring_containment(angdist2_p, 0.4 * u.deg, 0.3 * u.deg)
    # FIX: ring_radius and ring_halfwidth should have units of deg
    # FIX: hardcoded at the moment, but ring_radius should be read from
    # the gamma file (point-like) or given as input (diffuse).
    # FIX: ring_halfwidth should be given as input

    # Initialize arrays

    final_gammas = np.ndarray(shape=(n_bins_energy))
    final_protons = np.ndarray(shape=(n_bins_energy))
    pre_gammas = np.ndarray(shape=(n_bins_energy))
    pre_protons = np.ndarray(shape=(n_bins_energy))
    weighted_gamma_per_ebin = np.ndarray(n_bins_energy)
    weighted_proton_per_ebin = np.ndarray(n_bins_energy)
    sensitivity = np.ndarray(shape = n_bins_energy)
    n_excesses_min = np.ndarray(shape = n_bins_energy)
    eff_g = np.ndarray(shape = n_bins_energy)
    eff_p = np.ndarray(shape = n_bins_energy)
    gcut = np.ndarray(shape = n_bins_energy)
    tcut = np.ndarray(shape = n_bins_energy)
    gamma_rate = np.ndarray(shape = n_bins_energy)
    proton_rate = np.ndarray(shape = n_bins_energy)

    #Total rate of gammas and protons
    total_rate_proton = events_p.shape[0]/obstime_real
    total_rate_gamma = np.sum(rate_weighted_g)

    print("Total rate triggered proton {:.3f} Hz".format(total_rate_proton))
    print("Total rate triggered gamma  {:.3f} Hz".format(total_rate_gamma))

    #Dataframe to store the events which survive the cuts
    gammalike_events = pd.DataFrame(columns=events_g.keys())

    # Weight events and count number of events per bin:
    for i in range(0, n_bins_energy):  # binning in energy

        print("\n******** Energy bin: {:.3f} - {:.3f} TeV ********".format(energy[i].value, energy[i + 1].value))
        total_rate_proton_ebin = e_reco_p[(e_reco_p < energy[i+1]) & (e_reco_p > energy[i])].shape[0]/obstime_real
        total_rate_gamma_ebin = np.sum(rate_weighted_g[(e_reco_g < energy[i+1]) & (e_reco_g > energy[i])])

        #print("**************")
        print("Total rate triggered proton in this bin {:.5f} Hz".format(total_rate_proton_ebin.value))
        print("Total rate triggered gamma in this bin {:.5f} Hz".format(total_rate_gamma_ebin.value))

        #Calculate the cuts in gammaness and theta2 based on efficiency of weighted gammas

        rates_g = rate_weighted_g[(e_reco_g < energy[i+1]) & (e_reco_g > energy[i])]
        events_bin_g = events_g[(e_reco_g < energy[i+1]) & (e_reco_g > energy[i])]
        events_bin_p = events_p[(e_reco_p < energy[i+1]) & (e_reco_p > energy[i])]

        best_g_cut = find_cut(events_bin_g, rates_g, obstime,  "gammaness", 0.0, 1.0, gamma_eff_gammaness)

        events_g_after_g_cut=events_bin_g[events_bin_g.gammaness > best_g_cut]
        rates_g_after_g_cut=rates_g[events_bin_g.gammaness > best_g_cut]

        best_theta2_cut = find_cut(events_g_after_g_cut, rates_g_after_g_cut, obstime, "theta2", 0.0, .5, gamma_eff_theta2) * u.deg**2

        events_bin_after_cuts_g = events_bin_g[(events_bin_g.gammaness > best_g_cut) &(events_bin_g.theta2 < best_theta2_cut)]

        events_bin_after_cuts_p = events_p[(e_reco_p < energy[i+1]) & (e_reco_p > energy[i]) & \
                                           (gammaness_p > best_g_cut) & p_contained]




        #Save the survived events in the dataframe
        gammalike_events = pd.concat((gammalike_events, events_bin_after_cuts_g))
        gammalike_events = pd.concat((gammalike_events, events_bin_after_cuts_p))


        # ratio between the area where we search for protons ang_area_p
        # and the area where we search for gammas math.pi * t
        area_ratio_p = np.pi * best_theta2_cut / ang_area_p

        rate_g_ebin = np.sum(rate_weighted_g[(e_reco_g < energy[i+1]) & (e_reco_g > energy[i]) \
                                             & (gammaness_g > best_g_cut) & (theta2_g < best_theta2_cut)])

        rate_p_ebin = events_p[(e_reco_p < energy[i+1]) & (e_reco_p > energy[i]) \
                               & (gammaness_p > best_g_cut) & p_contained].shape[0]/obstime_real

        gamma_rate[i] = rate_g_ebin.to(1/u.min).to_value()
        proton_rate[i] = rate_p_ebin.to(1/u.min).to_value()*area_ratio_p

        final_gammas[i] = rate_g_ebin * obstime
        final_protons[i] = rate_p_ebin * obstime * area_ratio_p
        #print(area_ratio_p)

        pre_gammas[i] = e_reco_g[(e_reco_g < energy[i+1]) & (e_reco_g > energy[i]) \
                                   & (gammaness_g > best_g_cut) & (theta2_g < best_theta2_cut)].shape[0]
        pre_protons[i] = e_reco_p[(e_reco_p < energy[i+1]) & (e_reco_p > energy[i]) \
                                     & (gammaness_p > best_g_cut) & p_contained].shape[0]

        weighted_gamma_per_ebin[i] = np.sum(rate_weighted_g[(e_reco_g < energy[i+1]) & \
                                                    (e_reco_g > energy[i])]) * obstime
        weighted_proton_per_ebin[i] = events_bin_p.shape[0]


        gcut[i] = best_g_cut
        tcut[i] = best_theta2_cut.to_value()


        eff_g[i] = final_gammas[i] / weighted_gamma_per_ebin[i]
        eff_p[i] = final_protons[i] / weighted_proton_per_ebin[i]

    #n_excesses_min, sensitivity = calculate_sensitivity_lima(final_gammas, final_protons*noff,
    #                                                        1/noff * np.ones_like(final_gammas))
    n_excesses_min, sensitivity = calculate_sensitivity_lima(final_gammas, final_protons*noff,
                                                             1/noff * np.ones_like(final_gammas))

    # Avoid bins which are empty or have too few events:
    min_num_events = 10
    min_pre_events = 5

    # Set conditions for calculating sensitivity

    conditions = (
         (sensitivity<=0)
        | (pre_gammas<min_pre_events)
        | (pre_protons==0)
        | (final_gammas<min_num_events)
    )

    sensitivity[conditions] = np.inf

    # Compute sensitivity in flux units
    egeom = np.sqrt(energy[1:] * energy[:-1])
    dFdE, par = crab_hegra(egeom)

    sensitivity_flux = sensitivity / 100 * (dFdE * egeom * egeom).to(u.TeV / (u.cm**2 * u.s))

    print("\n******** Energy [TeV] *********\n")
    print(egeom)
    print("\nsensitivity flux:\n", sensitivity_flux)
    print("\nsensitivity[%]:\n", sensitivity)
    print("\n**************\n")


    list_of_tuples = list(zip(energy[:energy.shape[0]-1].to_value(), energy[1:].to_value(), gcut, tcut,
                            final_gammas, final_protons,
                            gamma_rate, proton_rate,
                              n_excesses_min, sensitivity, sensitivity_flux.to_value(),
                            eff_g, eff_p, pre_gammas, pre_protons))

    result = pd.DataFrame(list_of_tuples,
                           columns=['ebin_low', 'ebin_up', 'gammaness_cut', 'theta2_cut',
                                    'gammas_reweighted', 'protons_reweighted',
                                    'gamma_rate', 'proton_rate',
                                    'n_excesses_min', 'relative_sensitivity', 'sensitivity_flux',
                                    'eff_gamma', 'eff_proton',
                                    'mc_gammas', 'mc_protons'])

    return energy, sensitivity, result, gammalike_events, gcut, tcut


def sensitivity_gamma_efficiency_real_data(dl2_file_on, dl2_file_off,
                                           gcut, tcut,
                                           n_bins_energy,
                                           energy,
                                           gamma_eff_gammaness,
                                           gamma_eff_theta2,
                                           noff,
                                           obstime = 50 * 3600 * u.s):

    """
    Main function to calculate the sensitivity for cuts based
    on gamma efficiency using real data as ON and OFF events

    Parameters
    ---------
    dl2_file_g: `string` path to h5 file of ON events
    dl2_file_p: `string' path to h5 file of OFF events
    ntelescopes_gammas: `int` number of telescopes used
    ntelescopes_protons: `int` number of telescopes used
    n_bins_energy: `int` number of bins in energy
    gamma_eff_gammaness: `float` between 0 and 1 %/100
    of gammas to be left after cut in gammaness
    gamma_eff_theta2: `float` between 0 and 1 %/100
    of gammas to be left after cut in theta2
    noff: `float` ratio between the background and the signal region
    obstime: `Quantity` Observation time in seconds

    Returns
    ---------
    energy: `array` center of energy bins
    sensitivity: `array` sensitivity per energy bin

    """

    gammaness_on, theta2_on, e_reco_on, events_on, obstime_on = process_real(dl2_file_on)
    gammaness_off, angdist2_off, e_reco_off, events_off, obstime_off = process_real(dl2_file_off)

    #obstime_on = 6846.0 *u.s
    #obstime_off = 4188.0 *u.s

    # Extract spectral parameters
    print(energy)
    dFdE, crab_par = crab_hegra(energy)

    #For background, select protons contained in a ring overlapping with the ON region
    #p_contained, ang_area_p = ring_containment(angdist2_off, 0.6 * u.deg, 0.6 * u.deg)

    # Initialize arrays

    final_on = np.ndarray(shape=(n_bins_energy))
    final_off = np.ndarray(shape=(n_bins_energy))
    pre_on = np.ndarray(shape=(n_bins_energy))
    pre_off = np.ndarray(shape=(n_bins_energy))
    weighted_on_per_ebin = np.ndarray(n_bins_energy)
    weighted_off_per_ebin = np.ndarray(n_bins_energy)
    sensitivity = np.ndarray(shape = n_bins_energy)
    n_excesses_min = np.ndarray(shape = n_bins_energy)
    eff_on = np.ndarray(shape = n_bins_energy)
    eff_off = np.ndarray(shape = n_bins_energy)
    on_rate = np.ndarray(shape = n_bins_energy)
    off_rate = np.ndarray(shape = n_bins_energy)

    #Total rate of on and off data
    total_rate_off = events_off.shape[0]/obstime_off
    total_rate_on = events_on.shape[0]/obstime_on
    print("Total rate triggered OFF events {:.3f} Hz".format(total_rate_off))
    print("Total rate triggered ON events  {:.3f} Hz".format(total_rate_on))

    #Dataframe to store the events which survive the cuts
    gammalike_events = pd.DataFrame(columns=events_on.keys())

    for i in range(0, n_bins_energy):  # binning in energy

        print("\n******** Energy bin: {:.3f} - {:.3f} TeV ********".format(energy[i].value, energy[i + 1].value))
        total_rate_off_ebin = e_reco_off[(e_reco_off < energy[i+1]) & (e_reco_off > energy[i])].shape[0]/obstime_off
        total_rate_on_ebin = e_reco_on[(e_reco_on < energy[i+1]) & (e_reco_on > energy[i])].shape[0]/obstime_on


        #print("**************")
        print("Total rate triggered off events in this bin {:.5f} Hz".format(total_rate_off_ebin.value))
        print("Total rate triggered on events in this bin {:.5f} Hz".format(total_rate_on_ebin.value))

        #Calculate the cuts in gammaness and theta2 based on efficiency of weighted gammas

        events_bin_on = events_on[(e_reco_on < energy[i+1]) & (e_reco_on > energy[i])]

        events_bin_off = events_off[(e_reco_off < energy[i+1]) & (e_reco_off > energy[i])]

        best_g_cut = gcut[i]#find_cut(events_bin_on, 1, obstime,  "gammaness", 0, 1.0, gamma_eff_gammaness, True)



        events_on_after_g_cut=events_bin_on[events_bin_on.gammaness > best_g_cut]
        events_off_after_g_cut=events_bin_on[events_bin_on.gammaness > best_g_cut]

        best_theta2_cut = tcut[i]#find_cut_real(events_on_after_g_cut, events_off_after_g_cut, obstime_on, obstime_off, "theta2", 0.0, 1.0, gamma_eff_theta2) * u.deg**2
        #tcut[i]=best_theta2_cut.to_value()
        best_theta2_cut_off=0.5 #* u.deg**2

        events_bin_after_cuts_on = events_bin_on[(events_bin_on.gammaness > best_g_cut) & \
                                                 (events_bin_on.theta2 < best_theta2_cut)]

        events_bin_after_cuts_off = events_bin_off[(events_bin_off.gammaness > best_g_cut) & \
                                                   (events_bin_off.theta2 < best_theta2_cut_off)]


        #Save the survived events in the dataframe
        gammalike_events = pd.concat((gammalike_events, events_bin_after_cuts_on))
        gammalike_events = pd.concat((gammalike_events, events_bin_after_cuts_off))

        ang_area_p = np.pi * best_theta2_cut_off
        area_ratio_p = np.pi * best_theta2_cut / ang_area_p

        rate_off_ebin = events_off[(e_reco_off < energy[i+1]) & (e_reco_off > energy[i]) \
                                 & (gammaness_off > best_g_cut) & \
                                   (events_bin_off.theta2 < best_theta2_cut_off)].shape[0]/obstime_off

        rate_on_ebin = events_on[(e_reco_on < energy[i+1]) & (e_reco_on > energy[i]) \
                                 & (gammaness_on > best_g_cut) & \
                                 (events_bin_on.theta2 < best_theta2_cut)].shape[0]/obstime_on

        on_rate[i] = rate_on_ebin.to(1/u.min).to_value()
        off_rate[i] = rate_off_ebin.to(1/u.min).to_value() * area_ratio_p

        final_on[i] = rate_on_ebin * obstime
        final_off[i] = rate_off_ebin * obstime * area_ratio_p

        pre_off[i] = e_reco_off[(e_reco_off < energy[i+1]) & (e_reco_off > energy[i]) \
                                  & (gammaness_off > best_g_cut) & (events_bin_off.theta2 < best_theta2_cut_off)].shape[0]
        pre_on[i] = e_reco_on[(e_reco_on < energy[i+1]) & (e_reco_on > energy[i]) \
                              & (gammaness_on > best_g_cut) & \
                              (events_bin_on.theta2 < best_theta2_cut)].shape[0]


        print(on_rate[i], off_rate[i])
        print(final_on[i], final_off[i])
        print(pre_on[i], pre_off[i])

        eff_on[i] = pre_on[i] / events_bin_on.shape[0]
        eff_off[i] = pre_off[i] / events_bin_off.shape[0]

    signal = final_on - final_off

    rate_gammas = (signal/obstime).to(1/u.min).to_value()

    n_excesses_min, sensitivity = calculate_sensitivity_lima(signal, final_off*noff,
    1/noff* np.ones_like(final_on))

        # Avoid bins which are empty or have too few events:
    min_num_events = 10
    min_pre_events = 5

    # Set conditions for calculating sensitivity

    conditions = (
         (sensitivity<=0)
        | (pre_on<min_pre_events)
        | (pre_on==0)
        | (final_on<min_num_events)
    )

    sensitivity[conditions] = np.inf

    # Compute sensitivity in flux units
    egeom = np.sqrt(energy[1:] * energy[:-1])
    dFdE, par = crab_hegra(egeom)
    sensitivity_flux = sensitivity / 100 * (dFdE * egeom * egeom).to(u.TeV / (u.cm**2 * u.s))

    print("\n******** Energy [TeV] *********\n")
    print(egeom)
    print("\nsensitivity flux:\n", sensitivity_flux)
    print("\nsensitivity[%]:\n", sensitivity)
    print("\n**************\n")

    list_of_tuples = list(zip(energy[:energy.shape[0]-1].to_value(), energy[1:].to_value(), gcut, tcut,
                              final_on, final_off,
                              rate_gammas, off_rate,
                              n_excesses_min, sensitivity, sensitivity_flux.to_value(),
                              eff_on, eff_off, pre_on, pre_off))

    result = pd.DataFrame(list_of_tuples,
                           columns=['ebin_low', 'ebin_up', 'gammaness_cut', 'theta2_cut',
                                    'gammas_reweighted', 'protons_reweighted',
                                    'gamma_rate', 'proton_rate',
                                    'n_excesses_min', 'relative_sensitivity', 'sensitivity_flux',
                                    'eff_gamma', 'eff_proton',
                                    'mc_gammas', 'mc_protons'])

    return energy, sensitivity, result, gammalike_events, gcut, tcut
