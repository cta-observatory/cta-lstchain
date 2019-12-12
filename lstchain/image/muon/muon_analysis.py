import numpy as np
from ctapipe.image.muon.features import ring_containment 
from ctapipe.image.muon.features import ring_completeness
from ctapipe.image.muon.features import npix_above_threshold
from ctapipe.image.muon.features import npix_composing_ring
from ctapipe.image.muon.muon_integrator import MuonLineIntegrate
from ctapipe.image.cleaning import tailcuts_clean

from astropy.coordinates import Angle, SkyCoord, AltAz
from ctapipe.image.muon.muon_ring_finder import ChaudhuriKunduRingFitter
from ctapipe.coordinates import CameraFrame, NominalFrame
from astropy import units as u

from lstchain.image.muon import plot_muon_event
import matplotlib.pyplot as plt

__all__ = ['get_muon_center',
           'fit_muon',
           'analyze_muon_event',
           'muon_filter',
           'tag_pix_thr',
           ]

def get_muon_center(geom, equivalent_focal_length):
    """
    Get the x,y coordinates of the center of the muon ring
    in the NominalFrame

    Paramenters
    ---------
    geom: CameraGeometry
    equivalent_focal_length:    Focal length of the telescope

    Returns
    ---------
    x, y:    `floats` coordinates in  the NominalFrame
    """

    x, y = geom.pix_x, geom.pix_y

    telescope_pointing = SkyCoord(
            alt = 70 * u.deg,
            az = 0 * u.deg,
            frame = AltAz()
        )

    camera_coord = SkyCoord(
            x = x, y = y,
            frame = CameraFrame(
                focal_length = equivalent_focal_length,
                rotation = geom.pix_rotation,
                telescope_pointing = telescope_pointing
            )
    )
    nom_coord = camera_coord.transform_to(
            NominalFrame(origin=telescope_pointing)
        )

    x = nom_coord.delta_az.to(u.deg)
    y = nom_coord.delta_alt.to(u.deg)

    return x, y

def fit_muon(x, y, image, geom, tailcuts):
    """
    Fit the muon ring

    Paramenters
    ---------
    x, y:    `floats` coordinates in  the NominalFrame
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

    muonring = ChaudhuriKunduRingFitter(None)
    clean_mask = tailcuts_clean(geom, image, picture_thresh=tailcuts[0],
                                    boundary_thresh=tailcuts[1])
    image_clean = image * clean_mask
    muonringparam = muonring.fit(x, y, image_clean)

    dist = np.sqrt(np.power(x - muonringparam.ring_center_x, 2)
                       + np.power(y - muonringparam.ring_center_y, 2))
    ring_dist = np.abs(dist - muonringparam.ring_radius)
    muonringparam = muonring.fit(
            x, y, image_clean * (ring_dist < muonringparam.ring_radius * 0.4)
        )

    dist = np.sqrt(np.power(x - muonringparam.ring_center_x, 2) +
                       np.power(y - muonringparam.ring_center_y, 2))
    ring_dist = np.abs(dist - muonringparam.ring_radius)

    muonringparam = muonring.fit(
            x, y, image_clean * (ring_dist < muonringparam.ring_radius * 0.4)
        )
    
    return muonringparam, clean_mask, dist, image_clean

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
    min_pix = 148  # 8%

    x, y = get_muon_center(geom, equivalent_focal_length)
    muonringparam, clean_mask, dist, image_clean = fit_muon(x, y, image, geom, tailcuts)

    mirror_radius = np.sqrt(mirror_area / np.pi)
    dist_mask = np.abs(dist - muonringparam.ring_radius
                    ) < muonringparam.ring_radius * 0.4
    pix_ring = image * dist_mask
    pix_out_ring = image * ~dist_mask

    nom_dist = np.sqrt(np.power(muonringparam.ring_center_x,2) 
                    + np.power(muonringparam.ring_center_y, 2))

    muonringparam.ring_containment = ring_containment(
            muonringparam.ring_radius,
            cam_rad,
            muonringparam.ring_center_x,
            muonringparam.ring_center_y)

    ctel = MuonLineIntegrate(
                mirror_radius, hole_radius = 0.308 * u.m,
                pixel_width=0.1 * u.deg,
                sct_flag=False,
                secondary_radius = 0. * u.m
            )

    muonintensityoutput = ctel.fit_muon(muonringparam.ring_center_x,
                                    muonringparam.ring_center_y,
                                    muonringparam.ring_radius,
                                    x[dist_mask], y[dist_mask],
                                    image[dist_mask])
    muonintensityoutput.mask = dist_mask
    idx_ring = np.nonzero(pix_ring)
    muonintensityoutput.ring_completeness = ring_completeness(
                    x[idx_ring], y[idx_ring], pix_ring[idx_ring],
                    muonringparam.ring_radius,
                    muonringparam.ring_center_x,
                    muonringparam.ring_center_y,
                    threshold=30,
                    bins=30)
    muonintensityoutput.ring_size = np.sum(pix_ring)
    size_outside_ring = np.sum(pix_out_ring * clean_mask)
    dist_ringwidth_mask = np.abs(dist - muonringparam.ring_radius
                                             ) < (muonintensityoutput.ring_width)
    pix_ringwidth_im = image * dist_ringwidth_mask
    idx_ringwidth = np.nonzero(pix_ringwidth_im)

    muonintensityoutput.ring_pix_completeness = npix_above_threshold(
                    pix_ringwidth_im[idx_ringwidth], tailcuts[0]) / len(
                    pix_ring[idx_ringwidth])

    print("Impact parameter = %s"
                             "ring_width=%s, ring radius=%s, ring completeness=%s"% (
                             muonintensityoutput.impact_parameter,
                             muonintensityoutput.ring_width,
                             muonringparam.ring_radius,
                             muonintensityoutput.ring_completeness))

    conditions = [
        muonintensityoutput.impact_parameter <
        0.9 * mirror_radius,  # 90% inside the mirror
        
        muonintensityoutput.impact_parameter >
        0.2 * mirror_radius,  # 20% inside the mirror

        npix_above_threshold(pix_ring, tailcuts[0]) > 
        0.1 * min_pix,

        npix_composing_ring(pix_ring) >
        min_pix,
        
        muonringparam.ring_radius <
        1.5 * u.deg,

        muonringparam.ring_radius >
        1. * u.deg
        # TODO: To be applied when we have decent optics
        # muonintensityoutput.ring_width
        # < 0.08,
        
        # muonintensityoutput.ring_width
        # > 0.04
                ]

    muonintensityparam = muonintensityoutput 
    if all(conditions):
        good_ring = True
    else:
        good_ring = False

    if(plot_rings and plots_path and good_ring):
        altaz = AltAz(alt = 70 * u.deg, az = 0 * u.deg)
        focal_length = equivalent_focal_length        
        ring_nominal = SkyCoord(
                delta_az = muonringparam.ring_center_x,
                delta_alt = muonringparam.ring_center_y,
                frame = NominalFrame(origin=altaz)
            )

        ring_camcoord = ring_nominal.transform_to(CameraFrame(
                focal_length = focal_length,
                rotation = geom.pix_rotation,
                telescope_pointing = altaz))
        centroid = (ring_camcoord.x.value, ring_camcoord.y.value)
        radius = muonringparam.ring_radius
        width = muonintensityoutput.ring_width
        ringrad_camcoord = 2 * radius.to(u.rad) * focal_length
        ringwidthfrac = width / radius
        ringrad_inner = ringrad_camcoord * (1. - ringwidthfrac)
        ringrad_outer = ringrad_camcoord * (1. + ringwidthfrac)

        fig, ax = plt.subplots(figsize=(10,10))
        plot_muon_event(ax, geom, image * clean_mask, centroid, ringrad_camcoord, 
                        ringrad_inner, ringrad_outer, event_id)

        fig.savefig('{}/Event_{}_fitted.png'.format(plots_path, event_id))

    if(plot_rings and not plots_path):
        print("You are trying to plot without giving a path!")

    return muonintensityparam, size_outside_ring, muonringparam, good_ring

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
