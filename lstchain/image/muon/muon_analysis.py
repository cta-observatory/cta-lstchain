import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from ctapipe.containers import (
    MuonEfficiencyContainer,
    MuonParametersContainer,
)
from ctapipe.coordinates import (
    CameraFrame,
    TelescopeFrame,
)
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.image.muon import (
    MuonIntensityFitter,
    MuonRingFitter
)
from ctapipe.image.muon.features import (
    ring_completeness,
    ring_containment,
)

from lstchain.image.muon import plot_muon_event

__all__ = [
    'analyze_muon_event',
    'create_muon_table',
    'fill_muon_event',
    'fit_muon',
    'muon_filter',
    'pixel_coords_to_telescope',
    'radial_light_distribution',
    'tag_pix_thr',
    'update_parameters',
]


def pixel_coords_to_telescope(geom, equivalent_focal_length):
    """
    Get the x, y coordinates of the pixels in the telescope frame

    Parameters
    ----------
    geom : `CameraGeometry`
        Camera geometry
    equivalent_focal_length: `float`
        Focal length of the telescope

    Returns
    -------
    fov_lon, fov_lat : `floats`
        Coordinates in  the TelescopeFrame
    """

    camera_coord = SkyCoord(geom.pix_x, geom.pix_y,
                            CameraFrame(focal_length=equivalent_focal_length,
                                        rotation=geom.cam_rotation))
    tel_coord = camera_coord.transform_to(TelescopeFrame())

    return tel_coord.fov_lon, tel_coord.fov_lat


def update_parameters(config, n_pixels):
    """
    Create the parameters used to select good muon rings and perform the muon analysis.

    Parameters
    ----------
    config: `dict` or None
        Subset of parameters to be updated
    n_pixels: `int`
        Number of pixels of the camera

    Returns
    -------
    params: `dict`
        Dictionary of parameters used for the muon analysis

    """
    params = {
        'tailcuts': [10, 5],  # Thresholds used for the tail_cut cleaning
        'min_pix': 0.08,  # minimum fraction of the number of pixels in the ring with >0 signal
        'min_pix_fraction_after_cleaning': 0.1,  # minimum fraction of the ring pixels that must be above tailcuts[0]
        'min_ring_radius': 0.8 * u.deg,  # minimum ring radius
        'max_ring_radius': 1.5 * u.deg,  # maximum ring radius
        'max_radial_stdev': 0.1 * u.deg,  # maximum standard deviation of the light distribution along ring radius
        'max_radial_excess_kurtosis': 1.,  # maximum excess kurtosis
        'min_impact_parameter': 0.2,  # in fraction of mirror radius
        'max_impact_parameter': 0.9,  # in fraction of mirror radius
        'ring_integration_width': 0.25,  # +/- integration range along ring radius,
                                         # in fraction of ring radius (was 0.4 until 20200326)
        'outer_ring_width': 0.2,  # in fraction of ring radius, width of ring just outside the
                                  # integrated muon ring, used to check pedestal bias
        'ring_completeness_threshold': 30,  # Threshold in p.e. for pixels used in the ring completeness estimation
    }
    if config is not None:
        for key in config.keys():
            params[key] = config[key]
    params['min_pix'] = int(n_pixels * params['min_pix'])

    return params


def fit_muon(x, y, image, geom, tailcuts):
    """
    Fit the muon ring

    Parameters
    ----------
    x, y : `floats`
        Coordinates in  the TelescopeFrame
    image : `np.ndarray`
        Number of photoelectrons in each pixel
    geom : CameraGeometry
    tailcuts :`list`
        Tail cuts for image cleaning

    Returns
    -------
    muonringparam
    clean_mask: `np.ndarray`
        Mask after cleaning
    dist: `np.ndarray`
        Distance of every pixel to the center of the muon ring
    image_clean: `np.ndarray`
        Image after cleaning
    """

    fitter = MuonRingFitter(fit_method='kundu_chaudhuri')

    clean_mask = tailcuts_clean(
        geom, image,
        picture_thresh=tailcuts[0],
        boundary_thresh=tailcuts[1],
    )

    ring = fitter(x, y, image, clean_mask)

    max_allowed_outliers_distance = 0.4

    # Do an iterative fit removing pixels which are beyond
    # max_allowed_outliers_distance * radius of the ring
    # (along the radial direction)
    # The goal is to improve fit for good rings
    # with very few additional non-ring bright pixels.
    for _ in (0, 0):  # just to iterate the fit twice more
        dist = np.sqrt(
            (x - ring.center_x) ** 2 + (y - ring.center_y) ** 2
        )
        ring_dist = np.abs(dist - ring.radius)

        clean_mask *= (ring_dist < ring.radius * max_allowed_outliers_distance)
        ring = fitter(x, y, image, clean_mask)

    image_clean = image * clean_mask
    return ring, clean_mask, dist, image_clean


def analyze_muon_event(subarray, tel_id, event_id, image, good_ring_config, plot_rings, plots_path):
    """
    Analyze an event to fit a muon ring

    Parameters
    ----------
    subarray: `ctapipe.instrument.subarray.SubarrayDescription`
        Telescopes subarray
    tel_id : `int`
        Id of the telescope used
    event_id : `int`
        Id of the analyzed event
    image : `np.ndarray`
        Number of photoelectrons in each pixel
    good_ring_config : `dict` or None
        Set of parameters used to identify good muon rings to update LST-1 defaults
    plot_rings : `bool`
        Plot the muon ring
    plots_path : `string`
        Path to store the figures

    Returns
    -------
    muonintensityoutput : `MuonEfficiencyContainer`
    dist_mask : `ndarray`
        Pixels used in ring intensity likelihood fit
    ring_size : `float`
        Total intensity in ring in photoelectrons
    size_outside_ring : `float`
        Intensity outside the muon ring in photoelectrons
        to check for "shower contamination"
    muonringparam : `MuonRingContainer`
    good_ring : `bool`
        It determines whether the ring can be used for analysis or not
    radial_distribution : `dict`
        Return of function radial_light_distribution
    mean_pixel_charge_around_ring : float
        Charge "just outside" ring, to check the possible signal extractor bias
    muonparameters : `MuonParametersContainer`
    """

    tel_description = subarray.tels[tel_id]

    cam_rad = (
                      tel_description.camera.geometry.guess_radius() / tel_description.optics.equivalent_focal_length
              ) * u.rad
    geom = tel_description.camera.geometry
    equivalent_focal_length = tel_description.optics.equivalent_focal_length
    mirror_area = tel_description.optics.mirror_area

    # some parameters for analysis and cuts for good ring selection:
    params = update_parameters(good_ring_config, geom.n_pixels)

    x, y = pixel_coords_to_telescope(geom, equivalent_focal_length)
    muonringparam, clean_mask, dist, image_clean = fit_muon(x, y, image, geom,
                                                            params['tailcuts'])

    mirror_radius = np.sqrt(mirror_area / np.pi)  # meters
    dist_mask = np.abs(dist - muonringparam.radius
                       ) < muonringparam.radius * params['ring_integration_width']
    pix_ring = image * dist_mask
    pix_outside_ring = image * ~dist_mask

    # mask to select pixels just outside the ring that will be integrated to obtain the ring's intensity:
    dist_mask_2 = np.logical_and(~dist_mask,
                                 np.abs(dist - muonringparam.radius) <
                                 muonringparam.radius *
                                 (params['ring_integration_width'] + params['outer_ring_width']))
    pix_ring_2 = image[dist_mask_2]

    #    nom_dist = np.sqrt(np.power(muonringparam.center_x,2)
    #                    + np.power(muonringparam.center_y, 2))

    muonparameters = MuonParametersContainer()
    muonparameters.containment = ring_containment(
        muonringparam.radius,
        muonringparam.center_x, muonringparam.center_y, cam_rad)

    radial_distribution = radial_light_distribution(
        muonringparam.center_x,
        muonringparam.center_y,
        x[clean_mask], y[clean_mask],
        image[clean_mask])

    # Do complicated calculations (minuit-based max likelihood ring fit) only for selected rings:
    candidate_clean_ring = all(
        [radial_distribution['standard_dev'] < params['max_radial_stdev'],
         radial_distribution['excess_kurtosis'] < params['max_radial_excess_kurtosis'],
         (pix_ring > params['tailcuts'][0]).sum() >
         params['min_pix_fraction_after_cleaning'] * params['min_pix'],
         np.count_nonzero(pix_ring) > params['min_pix'],
         muonringparam.radius < params['max_ring_radius'],
         muonringparam.radius > params['min_ring_radius']
         ])

    if candidate_clean_ring:
        intensity_fitter = MuonIntensityFitter(subarray, hole_radius_m=0.308)

        # Use same hard-coded value for pedestal fluctuations as the previous
        # version of ctapipe:
        pedestal_stddev = 1.1 * np.ones(len(image))

        muonintensityoutput = \
            intensity_fitter(tel_id,
                             muonringparam.center_x,
                             muonringparam.center_y,
                             muonringparam.radius,
                             image,
                             pedestal_stddev,
                             dist_mask)

        dist_ringwidth_mask = np.abs(dist - muonringparam.radius) < \
                              muonintensityoutput.width

        # We do the calculation of the ring completeness (i.e. fraction of whole circle) using the pixels
        # within the "width" fitted using MuonIntensityFitter
        muonparameters.completeness = ring_completeness(
            x[dist_ringwidth_mask], y[dist_ringwidth_mask],
            image[dist_ringwidth_mask],
            muonringparam.radius,
            muonringparam.center_x,
            muonringparam.center_y,
            threshold=params['ring_completeness_threshold'],
            bins=30)

        # No longer existing in ctapipe 0.8:
        # pix_ringwidth_im = image[dist_ringwidth_mask]
        # muonintensityoutput.ring_pix_completeness =  \
        #     (pix_ringwidth_im > tailcuts[0]).sum() / len(pix_ringwidth_im)

    else:
        # just to have the default values with units:
        muonintensityoutput = MuonEfficiencyContainer()
        muonintensityoutput.width = u.Quantity(np.nan, u.deg)
        muonintensityoutput.impact = u.Quantity(np.nan, u.m)
        muonintensityoutput.impact_x = u.Quantity(np.nan, u.m)
        muonintensityoutput.impact_y = u.Quantity(np.nan, u.m)

    # muonintensityoutput.mask = dist_mask # no longer there in ctapipe 0.8
    ring_size = np.sum(pix_ring)
    size_outside_ring = np.sum(pix_outside_ring * clean_mask)

    # This is just mean charge per pixel in pixels just around the ring
    # (on the outer side):
    mean_pixel_charge_around_ring = np.sum(pix_ring_2) / len(pix_ring_2)

    if candidate_clean_ring:
        print("Impact parameter={:.3f}, ring_width={:.3f}, ring radius={:.3f}, "
              "ring completeness={:.3f}".format(
            muonintensityoutput.impact,
            muonintensityoutput.width,
            muonringparam.radius,
            muonparameters.completeness, ))
    # Now add the conditions based on the detailed muon ring fit:
    conditions = [
        candidate_clean_ring,
        muonintensityoutput.impact < params['max_impact_parameter'] * mirror_radius,
        muonintensityoutput.impact > params['min_impact_parameter'] * mirror_radius,

        # TODO: To be applied when we have decent optics.
        # muonintensityoutput.width
        # < 0.08,
        # NOTE: inside "candidate_clean_ring" cuts there is already a cut in
        # the std dev of light distribution along ring radius, which is also
        # a measure of the ring width

        # muonintensityoutput.width
        # > 0.04
    ]

    if all(conditions):
        good_ring = True
    else:
        good_ring = False

    if (plot_rings and plots_path and good_ring):
        focal_length = equivalent_focal_length
        ring_telescope = SkyCoord(muonringparam.center_x,
                                  muonringparam.center_y,
                                  TelescopeFrame())

        ring_camcoord = ring_telescope.transform_to(CameraFrame(
            focal_length=focal_length,
            rotation=geom.cam_rotation,
        ))
        centroid = (ring_camcoord.x.value, ring_camcoord.y.value)
        radius = muonringparam.radius
        width = muonintensityoutput.width
        ringrad_camcoord = 2 * radius.to(u.rad) * focal_length
        ringwidthfrac = width / radius
        ringrad_inner = ringrad_camcoord * (1. - ringwidthfrac)
        ringrad_outer = ringrad_camcoord * (1. + ringwidthfrac)

        fig, ax = plt.subplots(figsize=(10, 10))
        plot_muon_event(ax, geom, image * clean_mask, centroid,
                        ringrad_camcoord, ringrad_inner, ringrad_outer,
                        event_id)

        plt.figtext(0.15, 0.20, 'radial std dev: {0:.3f}'. \
                    format(radial_distribution['standard_dev']))
        plt.figtext(0.15, 0.18, 'radial excess kurtosis: {0:.3f}'. \
                    format(radial_distribution['excess_kurtosis']))
        plt.figtext(0.15, 0.16, 'fitted ring width: {0:.3f}'.format(width))
        plt.figtext(0.15, 0.14, 'ring completeness: {0:.3f}'. \
                    format(muonparameters.completeness))

        fig.savefig('{}/Event_{}_fitted.png'.format(plots_path, event_id))

    if (plot_rings and not plots_path):
        print("You are trying to plot without giving a path!")

    return muonintensityoutput, dist_mask, ring_size, size_outside_ring, \
           muonringparam, good_ring, radial_distribution, \
           mean_pixel_charge_around_ring, muonparameters


def muon_filter(image, thr_low=0, thr_up=1.e10):
    """
    Tag muon with a double threshold on the image photoelectron size
    Default values apply no tagging

    Parameters
    ----------
    image : `np.ndarray`
        Number of photoelectrons in each pixel
    thr_low : `float`
        Lower size threshold in photoelectrons
    thr_up : `float`
        Upper size threshold in photoelectrons

    Returns
    -------
    `bool`
        It determines whether a muon was tagged or not

    """
    return image.sum() > thr_low and image.sum() < thr_up


def tag_pix_thr(image, thr_low=50, thr_up=500, pe_thr=10):
    """
    Tag event with a double threshold on the number of pixels above 10 photoelectrons.
    Default values apply elimination of pedestal and calibration events

    Parameters
    ----------
    image : `np.ndarray`
        Number of photoelectrons in each pixel
    thr_low : `int`
        Lower threshold for number of pixel > 10 pe
    thr_up : `int`
        Upper threshold for number of pixel > 10 pe
    pe_thr : 'float'
        Minimum number of photoelectrons for a pixel to be counted

    Returns
    -------
    `bool`
        It determines whether an event is in the given nr of pixel range

    """

    return ((np.size(image[image > pe_thr]) < thr_up) and
            (np.size(image[image > pe_thr]) > thr_low))


def radial_light_distribution(center_x, center_y, pixel_x, pixel_y, image):
    """
    Calculate the radial distribution of the muon ring

    Parameters
    ----------
    center_x : `float`
        Center of muon ring in the field of view from circle fitting
    center_y : `float`
        Center of muon ring in the field of view from circle fitting
    pixel_x : `ndarray`
        X position of pixels in image
    pixel_y : `ndarray`
        Y position of pixel in image
    image : `ndarray`
        Amplitude of image pixels

    Returns
    -------
    standard_dev, skewness

    """

    if np.sum(image) == 0:
        return {'standard_dev': np.nan * u.deg, 'skewness': np.nan, 'excess_kurtosis': np.nan}

    # Convert everything to degrees:
    x0 = center_x.to_value(u.deg)
    y0 = center_y.to_value(u.deg)
    pix_x = pixel_x.to_value(u.deg)
    pix_y = pixel_y.to_value(u.deg)

    pix_r = np.sqrt((pix_x - x0) ** 2 + (pix_y - y0) ** 2)

    # mean, standard deviation & skewness of light distribution along ring radius.
    # ring_radius calculated elsewhere is approximately equal to "mean", but not
    # exactly, so we recalculate it here:
    mean = np.average(pix_r, weights=image)
    delta_r = pix_r - mean
    standard_dev = np.sqrt(np.average(delta_r ** 2, weights=image))
    skewness = np.average(delta_r ** 3, weights=image) / standard_dev ** 3
    excess_kurtosis = np.average(delta_r ** 4, weights=image) / standard_dev ** 4 - 3.

    return {'standard_dev': standard_dev * u.deg, 'skewness': skewness,
            'excess_kurtosis': excess_kurtosis}


def create_muon_table():
    """
    Create the empty dictionary to include the parameters
    of the fitted muon

    Parameters
    ----------
    None

    Returns
    -------
    `dict`
    """

    return {'event_id': [],
            'event_time': [],
            'mc_energy': [],
            'ring_size': [],
            'size_outside': [],
            'ring_center_x': [],
            'ring_center_y': [],
            'ring_radius': [],
            'ring_width': [],
            'good_ring': [],
            'muon_efficiency': [],
            'ring_containment': [],
            'ring_completeness': [],
            #  missing in ctapipe 0.8:
            # 'ring_pixel_completeness': [],
            'impact_parameter': [],
            'impact_x_array': [],
            'impact_y_array': [],
            'radial_stdev': [],  # Standard deviation of (cleaned) light distribution along ring radius
            'radial_skewness': [],  # Skewness of (cleaned) light distribution along ring radius
            'radial_excess_kurtosis': [],  # Excess kurtosis of (cleaned) light distribution along ring radius
            #  missing in ctapipe 0.8:
            'num_pixels_in_ring': [],  # pixels inside the integration area around the ring
            'mean_pixel_charge_around_ring': [],
            # Average pixel charge in pixels surrounding the outer part of the ring
            'hg_peak_sample': [],  # Peak sample of stacked HG waveforms of bright ring pixels
            'lg_peak_sample': [],  # Peak sample of stacked LG waveforms of bright ring pixels
            }


def fill_muon_event(mc_energy, output_parameters, good_ring, event_id,
                    event_time, muonintensityparam, dist_mask,
                    muonringparam, radial_distribution, size,
                    size_outside_ring, mean_pixel_charge_around_ring,
                    muonparameters, hg_peak_sample=np.nan, lg_peak_sample=np.nan):
    """
    Fill the dictionary with the parameters of a muon event

    Parameters
    ----------
    mc_energy: `float`
        Energy for simulated muons
    output_parameters: `dict`
        Empty dictionary to include the parameters
        of the fitted muon
    good_ring : `bool`
        It determines whether the ring can be used for analysis or not
    event_id : `int`
        Id of the analyzed event
    event_time: `float`
        Time of the event
    muonintensityparam: `MuonParametersContainer`
    dist_mask : `ndarray`
        Pixels used in ring intensity likelihood fit
    muonringparam : `MuonParametersContainer`
    radial_distribution : `dict`
        Return of function radial_light_distribution
    size : `float`
        Total intensity in ring in photoelectrons
    size_outside_ring : `float`
        Intensity outside the muon ting in photoelectrons
        to check for "shower contamination"
    mean_pixel_charge_around_ring : float
        Charge "just outside" ring, to check the possible signal extractor bias
    muonparameters : `MuonParametersContainer`
    hg_peak_sample: `np.ndarray`
        HG sample of the peak
    lg_peak_sample: `np.ndarray`
        LG sample of the peak

    Returns
    -------
    None

    """

    output_parameters['event_id'].append(event_id)
    output_parameters['event_time'].append(event_time)
    output_parameters['mc_energy'].append(mc_energy)

    output_parameters['ring_size'].append(size)
    output_parameters['size_outside'].append(size_outside_ring)
    output_parameters['ring_center_x'].append(muonringparam.center_x.value)
    output_parameters['ring_center_y'].append(muonringparam.center_y.value)
    output_parameters['ring_radius'].append(muonringparam.radius.value)
    output_parameters['ring_width'].append(muonintensityparam.width.value)
    output_parameters['good_ring'].append(good_ring)
    output_parameters['muon_efficiency'].append(muonintensityparam.optical_efficiency)
    output_parameters['ring_containment'].append(muonparameters.containment)
    output_parameters['ring_completeness'].append(muonparameters.completeness)
    #  missing in ctapipe 0.8:
    # output_parameters['ring_pixel_completeness'].append(muonintensityparam.ring_pix_completeness)
    output_parameters['impact_parameter'].append(muonintensityparam.impact.value)
    output_parameters['impact_x_array'].append(muonintensityparam.impact_x.value)
    output_parameters['impact_y_array'].append(muonintensityparam.impact_y.value)
    output_parameters['radial_stdev'].append(radial_distribution['standard_dev'].value)
    output_parameters['radial_skewness'].append(radial_distribution['skewness'])
    output_parameters['radial_excess_kurtosis'].append(radial_distribution['excess_kurtosis'])
    output_parameters['num_pixels_in_ring'].append(np.sum(dist_mask))
    output_parameters['mean_pixel_charge_around_ring'].append(mean_pixel_charge_around_ring)
    output_parameters['hg_peak_sample'].append(hg_peak_sample)
    output_parameters['lg_peak_sample'].append(lg_peak_sample)

    return
