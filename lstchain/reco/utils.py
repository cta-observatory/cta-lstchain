#!/usr/bin/env python3
"""Module with auxiliar functions:
Transform AltAz coordinates into Camera coordinates (This should be
implemented already in ctapipe but I haven't managed to find how to
do it)
Calculate source position from disp_norm distance.
Calculate disp_ distance from source position.

Usage:

"import utils"
"""

import numpy as np
from ctapipe.coordinates import HorizonFrame
from ctapipe.coordinates import NominalFrame
import astropy.units as u
from ..io.containers import DispContainer
from astropy.utils import deprecated

def alt_to_theta(alt):
    """Transforms altitude (angle from the horizon upwards) to theta
    (angle from z-axis) for simtel array coordinate systems                                  
    Parameters:
    -----------
    alt: float

    Returns:
    --------
    float: theta
    
    """
    
    return (90 * u.deg - alt).to(alt.unit)


def az_to_phi(az):
    """Transforms azimuth (angle from north towards east)                
    to phi (angle from x-axis towards y-axis)                            
    for simtel array coordinate systems                                  
    Parameters:
    -----------
    az: float
    
    Returns:
    --------
    az: float
    """
    return -az

        
def cal_cam_source_pos(mc_alt,mc_az,mc_alt_tel,mc_az_tel,focal_length):
    """Transform Alt-Az source position into Camera(x,y) coordinates
    source position. 
    
    Parameters:
    -----------
    mc_alt: float
    Alt coordinate of the event

    mc_az: float
    Az coordinate of the event

    mc_alt_tel: float
    Alt coordinate of the telescope pointing

    mc_az_tel: float
    Az coordinate of the telescope pointing

    focal_length: float
    Focal length of the telescope
    
    Returns:
    --------
    float: source_x1,

    float: source_x2
    """

    mc_alt = alt_to_theta(mc_alt*u.rad).value
    mc_az = az_to_phi(mc_az*u.rad).value
    mc_alt_tel = alt_to_theta(mc_alt_tel*u.rad).value
    mc_az_tel = az_to_phi(mc_az_tel*u.rad).value
        
    #Sines and cosines of direction angles
    cp = np.cos(mc_az)
    sp = np.sin(mc_az)
    ct = np.cos(mc_alt)
    st = np.sin(mc_alt)
    
     #Shower direction coordinates

    sourcex = st*cp
    sourcey = st*sp
    sourcez = ct

    #print(sourcex)

    source = np.array([sourcex,sourcey,sourcez])
    source=source.T
    
    #Rotation matrices towars the camera frame
    
    rot_Matrix = np.empty((0,3,3))
            
    alttel = mc_alt_tel
    aztel = mc_az_tel
    mat_Y = np.array([[np.cos(alttel),0,np.sin(alttel)],
                      [0,1,0], 
                      [-np.sin(alttel),0,np.cos(alttel)]]).T
        
        
    mat_Z = np.array([[np.cos(aztel),-np.sin(aztel),0],
                      [np.sin(aztel),np.cos(aztel),0],
                      [0,0,1]]).T
        
    rot_Matrix = np.matmul(mat_Y,mat_Z)
    
    res = np.einsum("...ji,...i",rot_Matrix,source)
    res = res.T
    
    source_x = -focal_length*res[0]/res[2]
    source_y = -focal_length*res[1]/res[2]
    return source_x, source_y


def disp_to_pos(disp_dx, disp_dy, cog_x, cog_y):
    """
    Calculates source position in camera coordinates(x,y) from the reconstructed disp
    
    Parameters:
    -----------
    disp: DispContainer
    cog_x: float
    Coordinate x of the center of gravity of Hillas ellipse
    cog_y: float
    Coordinate y of the center of gravity of Hillas ellipse

    Returns:
    --------
    (source_pos_x, source_pos_y)
    """
    source_pos_x = cog_x + disp_dx
    source_pos_y = cog_y + disp_dy
   
    return source_pos_x, source_pos_y



def guess_type(filename):
    """Guess the particle type from the filename

    Parameters
    ----------
    filename: str

    Returns
    -------
    str: 'gamma', 'proton', 'electron' or 'unknown'
    """
    particles = ['gamma', 'proton', 'electron']
    for p in particles:
        if p in filename:
            return p
    return 'unknown'


def particle_number(particle_name):
    """
    Return an integer coding the particle type
    'gamma'=0
    'proton'=1
    'electron'=2
    'muon'=3

    Parameters
    ----------
    particle_name: str

    Returns
    -------
    int
    """
    return {
        'gamma': 0,
        'proton': 1,
        'electron': 2,
        'muon': 3,
    }[particle_name]


def get_event_pos_in_camera(event, tel):
    """
    Return the position of the source in the camera frame
    Parameters
    ----------
    event: `ctapipe.io.containers.DataContainer`
    tel: `ctapipe.instruement.telescope.TelescopeDescription`

    Returns
    -------
    (x, y) (float, float): position in the camera
    """
    array_pointing = HorizonFrame(alt=event.mcheader.run_array_direction[1],
                                  az=event.mcheader.run_array_direction[0])
    event_direction = HorizonFrame(alt=event.mc.alt.to(u.rad),
                                   az=event.mc.az.to(u.rad))

    nom_frame = NominalFrame(array_direction=array_pointing,
                         pointing_direction=array_pointing)

    event_dir_nom = event_direction.transform_to(nom_frame)
    focal = tel.optics.equivalent_focal_length
    return focal * (event_dir_nom.x.to(u.rad).value, event_dir_nom.y.to(u.rad).value)

@deprecated('31/01/2019', message='Use disp_parameters')
def disp_norm(source_pos, hillas):
    """
    Deprecated, use disp_parameters.
    Compute the norm of the disp_norm vector

    Parameters
    ----------
    event: `ctapipe.io.container.MCEventContainer`
    tel: `ctapipe.instrument.TelescopeDescription`
    hillas: `ctapipe.io.container.HillasParametersContainer`

    Returns
    -------
    disp_norm: float
    """
    disp_norm = np.sqrt(((source_pos[0] - hillas.x) ** 2 + (source_pos[1] - hillas.y) ** 2).sum())
    return disp_norm


def reco_source_position_sky(cog_x, cog_y, disp_dx, disp_dy, focal_length, pointing_alt, pointing_az):
    """
    Compute the reconstructed source position in the sky

    Parameters
    ----------
    cog_x: `astropy.units.Quantity`
    cog_y: `astropy.units.Quantity`
    disp: DispContainer
    focal_length: `astropy.units.Quantity`
    pointing_alt: `astropy.units.Quantity`
    pointing_az: `astropy.units.Quantity`

    Returns
    -------

    """
    src_x, src_y = disp_to_pos(disp_dx, disp_dy, cog_x, cog_y)
    return camera_to_sky(src_x, src_y, focal_length, pointing_alt, pointing_az)


def camera_to_sky(pos_x, pos_y, focal, pointing_alt, pointing_az):
    """

    Parameters
    ----------
    pos_x: X coordinate in camera (distance)
    pos_y: Y coordinate in camera (distance)
    focal: telescope focal (distance)
    pointing_alt: pointing altitude in angle unit
    pointing_az: pointing altitude in angle unit

    Returns
    -------
    (alt, az)

    Example:
    --------
    import astropy.units as u
    import numpy as np
    x = np.array([1,0]) * u.m
    y = np.array([1,1]) * u.m

    """
    pointing_direction = HorizonFrame(alt=pointing_alt, az=pointing_az)

    source_pos_in_camera = NominalFrame(array_direction=pointing_direction,
                                        pointing_direction=pointing_direction,
                                        x=pos_x/focal * u.rad,
                                        y=pos_y/focal * u.rad,
                                        )

    return source_pos_in_camera.transform_to(pointing_direction)


def sky_to_camera(alt, az, focal, pointing_alt, pointing_az):
    """
    Coordinate transform from aky position (alt, az) (in angles) to camera coordinates (x, y) in distance
    Parameters
    ----------
    alt: astropy Quantity
    az: astropy Quantity
    focal: astropy Quantity
    pointing_alt: pointing altitude in angle unit
    pointing_az: pointing altitude in angle unit

    Returns
    -------

    """
    pointing_direction = HorizonFrame(alt=pointing_alt, az=pointing_az)

    event_direction = HorizonFrame(alt=alt, az=az)
    nom_frame = NominalFrame(array_direction=pointing_direction,
                             pointing_direction=pointing_direction)

    event_dir_nom = event_direction.transform_to(nom_frame)

    camera_pos = NominalFrame(pointing_direction=pointing_direction,
                              x=event_dir_nom.x.to(u.rad).value * focal,
                              y=event_dir_nom.y.to(u.rad).value * focal,
                              )

    # return focal * (event_dir_nom.x.to(u.rad).value, event_dir_nom.y.to(u.rad).value)
    return camera_pos

def source_side(source_pos_x, cog_x):
    """
    Compute on what side of the center of gravity the source is in the camera.

    Parameters
    ----------
    source_pos_x: X coordinate of the source in the camera, float
    cog_x: X coordinate of the center of gravity, float

    Returns
    -------
    float: -1 or +1
    """
    return np.sign(source_pos_x - cog_x)


def source_dx_dy(source_pos_x, source_pos_y, cog_x, cog_y):
    """
    Compute the coordinates of the vector (dx, dy) from the center of gravity to the source position

    Parameters
    ----------
    source_pos_x: X coordinate of the source in the camera
    source_pos_y: Y coordinate of the source in the camera
    cog_x: X coordinate of the center of gravity in the camera
    cog_y: Y coordinate of the center of gravity in the camera

    Returns
    -------
    (dx, dy)
    """
    return source_pos_x - cog_x, source_pos_y - cog_y

def disp_vector(disp_norm, disp_angle, disp_sign):
    """
    Compute `disp_norm.dx` and `disp_norm.dy` vector from `disp_norm.norm`, `disp_norm.angle` and `disp_norm.sign`

    Parameters
    ----------
    disp_norm: float
    disp_angle: float
    disp_sign: float

    Returns
    -------
    disp_dx, disp_dy
    """
    return polar_to_cartesian(disp_norm, disp_angle, disp_sign)

def polar_to_cartesian(norm, angle, sign):
    """
    Polar to cartesian transformation.
    Angle is supposed to be included in [-pi/2:pi/2].

    Parameters
    ----------
    norm: float or `numpy.ndarray`
    angle: float or `numpy.ndarray`
    sign: float or `numpy.ndarray`

    Returns
    -------

    """
    assert np.isfinite([norm, angle, sign]).all()
    x = norm * sign * np.cos(angle)
    y = norm * sign * np.sin(angle)
    return x, y

def cartesian_to_polar(x, y):
    """
    Cartesian to polar transformation

    Parameters
    ----------
    x: float or `numpy.ndarray`
    y: float or `numpy.ndarray`

    Returns
    -------
    norm, angle, sign
    """
    norm = np.sqrt(x**2 + y**2)
    if x == 0:
        angle = np.pi/2 * sign
    else:
        angle = np.arctan(np.tan(y/x))
    sign = np.sign(x)
    return norm, angle, sign


def predict_source_position_in_camera(cog_x, cog_y, disp_dx, disp_dy):
    """
    Compute the source position in the camera frame

    Parameters
    ----------
    cog_x: float or `numpy.ndarray` - x coordinate of the center of gravity (hillas.x)
    cog_y: float or `numpy.ndarray` - y coordinate of the center of gravity (hillas.y)
    disp_dx: float or `numpy.ndarray`
    disp_dy: float or `numpy.ndarray`

    Returns
    -------
    source_pos_x, source_pos_y
    """
    reco_src_x = cog_x + disp_dx
    reco_src_y = cog_y + disp_dy
    return reco_src_x, reco_src_y


def disp_parameters(hillas, source_pos_x, source_pos_y):
    """
    Compute the disp_norm parameters from Hillas parameters in the event position in the camera frame
    Return a `DispContainer`

    Parameters
    ----------
    hillas: `ctapipe.io.containers.HillasParametersContainer`
    source_pos_x: X coordinate of the source (event) position in the camera frame
    source_pos_y: Y coordinate of the source (event) position in the camera frame

    Returns
    -------
    `lstchain.io.containers.DispContainer`
    """
    disp = DispContainer()
    disp.dx = source_pos_x - hillas.x
    disp.dy = source_pos_y - hillas.y
    disp.norm = np.sqrt(disp.dx**2 + disp.dy**2)
    if disp.dx==0:
        disp.angle = np.pi/2. * np.sign(disp.dy)
    else:
        disp.angle = np.arctan(disp.dy/disp.dx)
    disp.sign = np.sign(disp.dx)
    disp.miss = np.abs(np.sin(hillas.psi.to(u.rad).value) * disp.dx - np.cos(hillas.psi.to(u.rad).value)*disp.dy)
    return disp
