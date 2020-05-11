import numpy as np
from ctapipe.image.muon.features import ring_containment
from ctapipe.image.muon.features import ring_completeness
from ctapipe.image.muon.features import npix_above_threshold
from ctapipe.image.muon.features import npix_composing_ring
#from ctapipe.image.muon.muon_integrator import MuonLineIntegrate
# Using provisionally a fixed version of MuonLineIntegrate, imported into
# lstchain! As soon as ctapipe 0.8 is out, we should go back to using the
# ctapipe version.
from lstchain.image.muon.muon_integrator import MuonLineIntegrate
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.io.containers import MuonIntensityParameter

from astropy.coordinates import SkyCoord, AltAz
from ctapipe.image.muon.muon_ring_finder import ChaudhuriKunduRingFitter
from ctapipe.coordinates import CameraFrame, TelescopeFrame
from astropy import units as u

from lstchain.image.muon import plot_muon_event
import matplotlib.pyplot as plt


__all__ = [
    'analyze_muon_event',
    'create_muon_table',
    'fill_muon_event',
    'fit_muon',
    'muon_filter',
    'pixel_coords_to_telescope',
    'radial_light_distribution',
    'tag_pix_thr',
]


def pixel_coords_to_telescope(geom, equivalent_focal_length):
    """
    Get the x, y coordinates of the pixels in the telescope frame

    Paramenters
    ---------
    geom: CameraGeometry
    equivalent_focal_length:    Focal length of the telescope

    Returns
    ---------
    delta_az, delta_alt:    `floats` coordinates in  the TelescopeFrame
    """

    camera_coord = SkyCoord(
        x=geom.pix_x,
        y=geom.pix_y,
        frame=CameraFrame(
            focal_length=equivalent_focal_length,
            rotation=geom.cam_rotation,
        )
    )
    tel_coord = camera_coord.transform_to(TelescopeFrame())

    return tel_coord.delta_az, tel_coord.delta_alt


def fit_muon(x, y, image, geom, tailcuts):
    """
    Fit the muon ring

    Paramenters
    ---------
    x, y:    `floats` coordinates in  the TelescopeFrame
    image:   `np.ndarray` number of photoelectrons in each pixel
    geom:    CameraGeometry
    image:   `list` tail cuts for image cleaning

    Returns
    ---------
    muonringparam: ``
    clean_mask:    `np.ndarray` mask after cleaning
    dist:          `np.ndarray` distance of every pixel
                    to the center of the muon ring
    image_clean:   `np.ndarray` image after cleaning
    """

    fitter = ChaudhuriKunduRingFitter()
    clean_mask = tailcuts_clean(
        geom, image,
        picture_thresh=tailcuts[0],
        boundary_thresh=tailcuts[1],
    )
    image_clean = image * clean_mask
    ring = fitter.fit(x, y, image_clean)

    max_allowed_outliers_distance = 0.4

    # Do an iterative fit removing pixels which are beyond
    # max_allowed_outliers_distance * ring_radius of the ring
    # (along the radial direction)
    # The goal is to improve fit for good rings
    # with very few additional non-ring bright pixels.
    for _ in (0, 0):  # just to iterate the fit twice more
        dist = np.sqrt(
            (x - ring.ring_center_x)**2 + (y - ring.ring_center_y)**2
        )
        ring_dist = np.abs(dist - ring.ring_radius)
        ring = fitter.fit(
            x, y,
            image_clean * (ring_dist < ring.ring_radius * max_allowed_outliers_distance)
        )

    return ring, clean_mask, dist, image_clean


def analyze_muon_event(event_id, image, geom, equivalent_focal_length,
                       mirror_area, plot_rings, plots_path):
    """
    Analyze an event to fit a muon ring

    Paramenters
    ---------
    event_id:   `int` id of the analyzed event
    image:      `np.ndarray` number of photoelectrons in each pixel
    geom:       CameraGeometry
    equivalent_focal_length: `float` focal length of the telescope
    mirror_area: `float` mirror area of the telescope
    plot_rings: `bool` plot the muon ring
    plots_path: `string` path to store the figures

    Returns
    ---------
    muonintensityoutput ``
    muonringparam       ``
    good_ring           `bool` it determines whether the ring can be used for analysis or not

    TODO: several hard-coded quantities that can go into a configuration file
    """

    tailcuts = [10, 5]

    cam_rad = 2.26 * u.deg

    # some cuts for good ring selection:
    min_pix = 148                              # (8%) minimum number of pixels in the ring with >0 signal
    min_pix_fraction_after_cleaning = 0.1      # minimum fraction of the ring pixels that must be above tailcuts[0]
    min_ring_radius = 0.8*u.deg                # minimum ring radius
    max_ring_radius = 1.5*u.deg                # maximum ring radius
    max_radial_stdev = 0.1*u.deg               # maximum standard deviation of the light distribution along ring radius
    max_radial_excess_kurtosis = 1.            # maximum excess kurtosis
    min_impact_parameter = 0.2                 # in fraction of mirror radius
    max_impact_parameter = 0.9                 # in fraction of mirror radius
    ring_integration_width = 0.25              # +/- integration range along ring radius, in fraction of ring radius (was 0.4 until 20200326)
    outer_ring_width = 0.2                     # in fraction of ring radius, width of ring just outside the integrated muon ring, used to check pedestal bias

    x, y = pixel_coords_to_telescope(geom, equivalent_focal_length)
    muonringparam, clean_mask, dist, image_clean = fit_muon(x, y, image, geom, tailcuts)

    mirror_radius = np.sqrt(mirror_area / np.pi)
    dist_mask = np.abs(dist - muonringparam.ring_radius
                    ) < muonringparam.ring_radius * ring_integration_width
    pix_ring = image * dist_mask
    pix_outside_ring = image * ~dist_mask

    # mask to select pixels just outside the ring that will be integrated to obtain the ring's intensity:
    dist_mask_2 = np.logical_and(~dist_mask, np.abs(dist-muonringparam.ring_radius) < muonringparam.ring_radius*(ring_integration_width+outer_ring_width))
    pix_ring_2 = image[dist_mask_2]

#    nom_dist = np.sqrt(np.power(muonringparam.ring_center_x,2)
#                    + np.power(muonringparam.ring_center_y, 2))

    muonringparam.ring_containment = ring_containment(
            muonringparam.ring_radius,
            cam_rad,
            muonringparam.ring_center_x,
            muonringparam.ring_center_y)

    radial_distribution = radial_light_distribution(
        muonringparam.ring_center_x,
        muonringparam.ring_center_y,
        x[clean_mask], y[clean_mask],
        image[clean_mask])


    # Do complicated calculations (minuit-based max likelihood ring fit) only for selected rings:
    candidate_clean_ring = all(
        [radial_distribution['standard_dev'] < max_radial_stdev,
         radial_distribution['excess_kurtosis'] < max_radial_excess_kurtosis,
         npix_above_threshold(pix_ring, tailcuts[0]) > min_pix_fraction_after_cleaning * min_pix,
         npix_composing_ring(pix_ring) > min_pix,
         muonringparam.ring_radius < max_ring_radius,
         muonringparam.ring_radius > min_ring_radius
        ])

    if  candidate_clean_ring:
        ctel = MuonLineIntegrate(
            mirror_radius, hole_radius = 0.308 * u.m,
            pixel_width=0.1 * u.deg,
            sct_flag=False,
            secondary_radius = 0. * u.m
        )

        muonintensityoutput = ctel.fit_muon(
            muonringparam.ring_center_x,
            muonringparam.ring_center_y,
            muonringparam.ring_radius,
            x[dist_mask], y[dist_mask],
            image[dist_mask])

        dist_ringwidth_mask = np.abs(dist - muonringparam.ring_radius) < (muonintensityoutput.ring_width)
        # We do the calculation of the ring completeness (i.e. fraction of whole circle) using the pixels
        # within the "width" fitted using MuonLineIntegrate.
        muonintensityoutput.ring_completeness = ring_completeness(
            x[dist_ringwidth_mask], y[dist_ringwidth_mask], image[dist_ringwidth_mask],
            muonringparam.ring_radius,
            muonringparam.ring_center_x,
            muonringparam.ring_center_y,
            threshold=30,
            bins=30)

        pix_ringwidth_im = image[dist_ringwidth_mask]
        muonintensityoutput.ring_pix_completeness =  npix_above_threshold(pix_ringwidth_im, tailcuts[0]) / len(pix_ringwidth_im)

    else:
            muonintensityoutput = MuonIntensityParameter()
            # Set default values for cases in which the muon intensity fit (with MuonLineIntegrate) is not done:
            muonintensityoutput.ring_width = np.nan*u.deg
            muonintensityoutput.impact_parameter_pos_x = np.nan*u.m
            muonintensityoutput.impact_parameter_pos_y = np.nan*u.m
            muonintensityoutput.impact_parameter = np.nan*u.m
            muonintensityoutput.ring_pix_completeness = np.nan
            muonintensityoutput.ring_completeness = np.nan


    muonintensityoutput.mask = dist_mask
    muonintensityoutput.ring_size = np.sum(pix_ring)
    size_outside_ring = np.sum(pix_outside_ring * clean_mask)

    # This is just mean charge per pixel in pixels just around the ring (on the outer side):
    mean_pixel_charge_around_ring = np.sum(pix_ring_2)/len(pix_ring_2)

    if candidate_clean_ring:
        print("Impact parameter={:.3f}, ring_width={:.3f}, ring radius={:.3f}, ring completeness={:.3f}".format(
            muonintensityoutput.impact_parameter, muonintensityoutput.ring_width,
            muonringparam.ring_radius, muonintensityoutput.ring_completeness,
))
    # Now add the conditions based on the detailed muon ring fit made by MuonLineIntegrate:
    conditions = [
        candidate_clean_ring,

        muonintensityoutput.impact_parameter <
        max_impact_parameter * mirror_radius,

        muonintensityoutput.impact_parameter >
        min_impact_parameter * mirror_radius,

        # TODO: To be applied when we have decent optics.
        # muonintensityoutput.ring_width
        # < 0.08,
        # NOTE: inside "candidate_clean_ring" cuts there is already a cut in the st dev of light distribution along ring radius,
        # which is also a measure of the ring width

        # muonintensityoutput.ring_width
        # > 0.04
    ]

    muonintensityparam = muonintensityoutput
    if all(conditions):
        good_ring = True
    else:
        good_ring = False

    if(plot_rings and plots_path and good_ring):
        focal_length = equivalent_focal_length
        ring_telescope = SkyCoord(
            delta_az=muonringparam.ring_center_x,
            delta_alt=muonringparam.ring_center_y,
            frame=TelescopeFrame()
        )

        ring_camcoord = ring_telescope.transform_to(CameraFrame(
            focal_length=focal_length,
            rotation=geom.cam_rotation,
        ))
        centroid = (ring_camcoord.x.value, ring_camcoord.y.value)
        radius = muonringparam.ring_radius
        width = muonintensityoutput.ring_width
        ringrad_camcoord = 2 * radius.to(u.rad) * focal_length
        ringwidthfrac = width / radius
        ringrad_inner = ringrad_camcoord * (1. - ringwidthfrac)
        ringrad_outer = ringrad_camcoord * (1. + ringwidthfrac)

        fig, ax = plt.subplots(figsize=(10, 10))
        plot_muon_event(ax, geom, image * clean_mask, centroid, ringrad_camcoord,
                        ringrad_inner, ringrad_outer, event_id)

        plt.figtext(0.15, 0.20, 'radial std dev: {0:.3f}'.format(radial_distribution['standard_dev']))
        plt.figtext(0.15, 0.18, 'radial excess kurtosis: {0:.3f}'.format(radial_distribution['excess_kurtosis']))
        plt.figtext(0.15, 0.16, 'fitted ring width: {0:.3f}'.format(width))
        plt.figtext(0.15, 0.14, 'ring completeness: {0:.3f}'.format(muonintensityoutput.ring_completeness))


        fig.savefig('{}/Event_{}_fitted.png'.format(plots_path, event_id))

    if(plot_rings and not plots_path):
        print("You are trying to plot without giving a path!")

    return muonintensityparam, size_outside_ring, muonringparam, good_ring, radial_distribution, mean_pixel_charge_around_ring

def muon_filter(image, thr_low = 0, thr_up = 1.e10):
    """
    Tag muon with a double threshold on the image photoelectron size
    Default values apply no tagging

    Paramenters
    ---------
    image:      `np.ndarray` number of photoelectrons in each pixel
    thr_low: `float` lower size threshold in photoelectrons
    thr_up: `float` upper size threshold in photoelectrons

    Returns
    ---------
    `bool` it determines whether a muon was tagged or not

    """
    return image.sum() > thr_low and image.sum() < thr_up

def tag_pix_thr(image, thr_low = 50, thr_up = 500, pe_thr = 10):
    """
    Tag event with a double threshold on the number of pixels above 10 photoelectrons
    Default values apply elimination of pedestal and calibration events

    Paramenters
    ---------
    image:      `np.ndarray` number of photoelectrons in each pixel
    thr_low: `int` lower threshold for number of pixel > 10 pe
    thr_up: `int` upper threshold for number of pixel > 10 pe
    pe_thr: 'float' minimum number of photoelectrons for a pixel to be counted

    Returns
    ---------
    `bool` it determines whether a the event is in the given nr of pixel range

    """

    return ((np.size(image[image > pe_thr]) < thr_up) and (np.size(image[image > pe_thr]) > thr_low))


def radial_light_distribution(center_x, center_y, pixel_x, pixel_y, image):
    """

    Parameters
    ----------
    center_x: float
        Center of muon ring in the field of view from circle fitting
    center_y: float
        Center of muon ring in the field of view from circle fitting
    pixel_x: ndarray
        X position of pixels in image
    pixel_y: ndarray
        Y position of pixel in image
    image: ndarray
        Amplitude of image pixels

    Returns
    -------
    standard_dev, skewness
    """

    if np.sum(image) == 0:
        return {'standard_dev' : np.nan*u.deg, 'skewness' : np.nan, 'excess_kurtosis' : np.nan}

    # Convert everything to degrees:
    x0 = center_x.to_value(u.deg)
    y0 = center_y.to_value(u.deg)
    pix_x = pixel_x.to_value(u.deg)
    pix_y = pixel_y.to_value(u.deg)

    pix_r = np.sqrt((pix_x-x0)**2 + (pix_y-y0)**2)

    # mean, standard deviation & skewness of light distribution along ring radius.
    # ring_radius calculated elsewhere is approximately equal to "mean", but not exactly, so we recalculate it here:
    mean = np.average(pix_r, weights=image)
    delta_r = pix_r - mean
    standard_dev = np.sqrt(np.average(delta_r**2, weights=image))
    skewness = np.average(delta_r**3, weights=image) / standard_dev**3
    excess_kurtosis = np.average(delta_r**4, weights=image)/standard_dev**4 - 3.

    return {'standard_dev' : standard_dev*u.deg, 'skewness' : skewness, 'excess_kurtosis' : excess_kurtosis}


def create_muon_table():

    return {'event_id': [],
            'event_time': [],
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
            'ring_pixel_completeness': [],
            'impact_parameter': [],
            'impact_x_array': [],
            'impact_y_array': [],
            'radial_stdev' : [],                  # Standard deviation of (cleaned) light distribution along ring radius
            'radial_skewness' : [],               # Skewness of (cleaned) light distribution along ring radius
            'radial_excess_kurtosis' : [],        # Excess kurtosis of (cleaned) light distribution along ring radius
            'num_pixels_in_ring' : [],            # pixels inside the integration area around the ring
            'mean_pixel_charge_around_ring' : [], # Average pixel charge in pixels surrounding the outer part of the ring
            'hg_peak_sample' : [],                # Peak sample of stacked HG waveforms of bright ring pixels
            'lg_peak_sample' : [],                # Peak sample of stacked LG waveforms of bright ring pixels
    }


def fill_muon_event(output_parameters, good_ring, event_id, event_time, muonintensityparam, muonringparam,
                    radial_distribution, size_outside_ring, mean_pixel_charge_around_ring,
                    hg_peak_sample=np.nan, lg_peak_sample=np.nan):

    output_parameters['event_id'].append(event_id)
    output_parameters['event_time'].append(event_time)
    output_parameters['ring_size'].append(muonintensityparam.ring_size)
    output_parameters['size_outside'].append(size_outside_ring)
    output_parameters['ring_center_x'].append(muonringparam.ring_center_x.value)
    output_parameters['ring_center_y'].append(muonringparam.ring_center_y.value)
    output_parameters['ring_radius'].append(muonringparam.ring_radius.value)
    output_parameters['ring_width'].append(muonintensityparam.ring_width.value)
    output_parameters['good_ring'].append(good_ring)
    output_parameters['muon_efficiency'].append(muonintensityparam.optical_efficiency_muon)
    output_parameters['ring_containment'].append(muonringparam.ring_containment)
    output_parameters['ring_completeness'].append(muonintensityparam.ring_completeness)
    output_parameters['ring_pixel_completeness'].append(muonintensityparam.ring_pix_completeness)
    output_parameters['impact_parameter'].append(muonintensityparam.impact_parameter.value)
    output_parameters['impact_x_array'].append(muonintensityparam.impact_parameter_pos_x.value)
    output_parameters['impact_y_array'].append(muonintensityparam.impact_parameter_pos_y.value)
    output_parameters['radial_stdev'].append(radial_distribution['standard_dev'].value)
    output_parameters['radial_skewness'].append(radial_distribution['skewness'])
    output_parameters['radial_excess_kurtosis'].append(radial_distribution['excess_kurtosis'])
    output_parameters['num_pixels_in_ring'].append(np.sum(muonintensityparam.mask))
    output_parameters['mean_pixel_charge_around_ring'].append(mean_pixel_charge_around_ring)
    output_parameters['hg_peak_sample'].append(hg_peak_sample)
    output_parameters['lg_peak_sample'].append(lg_peak_sample)
    return
