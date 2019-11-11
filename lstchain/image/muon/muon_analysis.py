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

__all__ = ['get_muon_center',
           'fit_muon',
           'analyze_muon_event',
           ]

def get_muon_center(geom):
    """
    Get the x,y coordinates of the center of the muon ring
    in the NominalFrame

    Paramenters
    ---------
    geom: CameraGeometry

    Returns
    ---------
    x, y:    `floats` coordinates in  the NominalFrame
    """

    x, y = geom.pix_x, geom.pix_y

    telescope_pointing = SkyCoord(
            alt=70 * u.deg,
            az=0 * u.deg,
            frame=AltAz()
        )

    camera_coord = SkyCoord(
            x=x, y=y,
            frame=CameraFrame(
                focal_length=teldes.optics.equivalent_focal_length,
                rotation=geom.pix_rotation,
                telescope_pointing=telescope_pointing
            )
    )
    nom_coord = camera_coord.transform_to(
            NominalFrame(origin=telescope_pointing)
        )


    x = nom_coord.delta_az.to(u.deg)
    y = nom_coord.delta_alt.to(u.deg)

    return x, y

def fit_muon(x, y, image, geom):
    """
    Fit the muon ring

    Paramenters
    ---------
    x, y:    `floats` coordinates in  the NominalFrame
    image:   `np.ndarray` number of photoelectrons in each pixel
    geom:    CameraGeometry

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

def analyze_muon_event(event_id, image, geom, plot_ring):
    """
    Analyze an event to fit a muon ring

    Paramenters
    ---------
    event_id:  `int` id of the analyzed event
    image:     `np.ndarray` number of photoelectrons in each pixel
    geom:      CameraGeometry
    plot_ring: `bool` plot the muon ring

    Returns
    ---------
    muonintensityoutput ``
    muonringparam       ``
    impact_condition    `bool` 

    TODO: several hard-coded quantities that can go into a configuration file
    """

    tailcuts = [5, 10]

    cam_rad = 2.26 * u.deg
    min_pix = 148  # 8%

    x, y = get_muon_center(geom)
    muonringparam, clean_mask, dist, image = fit_muon(x, y, image[0], geom)

    mir_rad = np.sqrt(teldes.optics.mirror_area.to("m2") / np.pi)
    dist_mask = np.abs(dist - muonringparam.ring_radius
                    ) < muonringparam.ring_radius * 0.4
    pix_im = image[0] * dist_mask
    nom_dist = np.sqrt(np.power(muonringparam.ring_center_x,2) 
                    + np.power(muonringparam.ring_center_y, 2))

    if(npix_above_threshold(pix_im, tailcuts[0]) > 0.1 * min_pix
       and npix_composing_ring(pix_im) > min_pix
       and nom_dist < cam_rad  
       and muonringparam.ring_radius < 1.5 * u.deg
       and muonringparam.ring_radius > 1. * u.deg):
        muonringparam.ring_containment = ring_containment(
            muonringparam.ring_radius,
            cam_rad,
            muonringparam.ring_center_x,
            muonringparam.ring_center_y)


    ctel = MuonLineIntegrate(
                mir_rad, hole_radius = 0.308 * u.m,
                pixel_width=0.1 * u.deg,
                sct_flag=False,
                secondary_radius = 0. * u.m
            )

    muonintensityoutput = ctel.fit_muon(muonringparam.ring_center_x,
                                    muonringparam.ring_center_y,
                                    muonringparam.ring_radius,
                                    x[dist_mask], y[dist_mask],
                                    image[0][dist_mask])
    muonintensityoutput.mask = dist_mask
    idx_ring = np.nonzero(pix_im)
    muonintensityoutput.ring_completeness = ring_completeness(
                    x[idx_ring], y[idx_ring], pix_im[idx_ring],
                    muonringparam.ring_radius,
                    muonringparam.ring_center_x,
                    muonringparam.ring_center_y,
                    threshold=30,
                    bins=30)
    muonintensityoutput.ring_size = np.sum(pix_im)
    dist_ringwidth_mask = np.abs(dist - muonringparam.ring_radius
                                             ) < (muonintensityoutput.ring_width)
    pix_ringwidth_im = image[0] * dist_ringwidth_mask
    idx_ringwidth = np.nonzero(pix_ringwidth_im)

    muonintensityoutput.ring_pix_completeness = npix_above_threshold(
                    pix_ringwidth_im[idx_ringwidth], tailcuts[0]) / len(
                    pix_im[idx_ringwidth])

    print("Impact parameter = %s"
                             "ring_width=%s, ring radius=%s, ring completeness=%s"% (
                             muonintensityoutput.impact_parameter,
                             muonintensityoutput.ring_width,
                             muonringparam.ring_radius,
                             muonintensityoutput.ring_completeness))

    conditions = [
        muonintensityoutput.impact_parameter <
        0.9 * mir_rad,  # 90% inside the mirror
        
        muonintensityoutput.impact_parameter >
        0.2 * mir_rad  # 20% inside the mirror
        
        # TODO: To be applied when we have decent optics
        # muonintensityoutput.ring_width
        # < 0.08,
        
        # muonintensityoutput.ring_width
        # > 0.04
                ]

    muonintensityparam = muonintensityoutput 
    if all(conditions):
        impact_condition = True
    else:
        impact_condition = False

    if(plot_ring):
        altaz = AltAz(alt = 70 * u.deg, az = 0 * u.deg)
        flen = event.inst.subarray.tel[0].optics.equivalent_focal_length
        ring_nominal = SkyCoord(
                delta_az=muonringparam.ring_center_x,
                delta_alt=muonringparam.ring_center_y,
                frame=NominalFrame(origin=altaz)
            )

        ring_camcoord = ring_nominal.transform_to(CameraFrame(
                focal_length=flen,
                rotation=geom.pix_rotation,
                telescope_pointing=altaz))
        centroid = (ring_camcoord.x.value, ring_camcoord.y.value)
        radius = muonringparam.ring_radius
        width = muonintensityoutput.ring_width
        ringrad_camcoord = 2 * radius.to(u.rad) * flen
        ringwidthfrac = width / radius
        ringrad_inner = ringrad_camcoord * (1. - ringwidthfrac)
        ringrad_outer = ringrad_camcoord * (1. + ringwidthfrac)

        fig, ax = plt.subplots(figsize=(10,10))
        plot_event(ax, geom, image, centroid, ringrad_camcoord, ringrad_inner, ringrad_outer,
                   event_id)

        fig.savefig('figures/Event_{}_fitted.png'.format(event_id))


    return muonintensityparam, muonringparam, impact_condition
