#!/usr/bin/env python3
"""Module with auxiliar functions:
Transform AltAz coordinates into Camera coordinates (This should be
implemented already in ctapipe but I haven't managed to find how to
do it)
Calculate source position from disp distance.
Calculate disp_ distance from source position.

Usage:

"import utils"
"""

import numpy as np
from ctapipe.coordinates import HorizonFrame
from ctapipe.coordinates import NominalFrame
import astropy.units as u

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

def calc_disp(source_x,source_y,cen_x,cen_y):
    """
    Calculates "disp" distance from source position in camera
    coordinates
    
    Parameters:
    -----------
    source_x: float
    Source coordinate X in camera frame

    source_y: float
    Source coordinate Y in camera frame

    cen_x = float
    Coordinate x of the center of gravity of Hillas ellipse

    cen_y = float
    Coordinate y of the center of gravity of Hillas ellipse

    Returns:
    --------
    float: disp
    """
    disp = np.sqrt((source_x-cen_x)**2
                   +(source_y-cen_y)**2)
    return disp

def disp_to_pos(disp,cen_x,cen_y,psi):
    """
    Calculates source position in camera coordinates(x,y) from "disp_"
    distance.
    For now, it only works for POINT GAMMAS, it doesn't take into
    account the duplicity of the disp method.
    
    Parameters:
    -----------
    disp: float - disp distance

    cen_x = float
    Coordinate x of the center of gravity of Hillas ellipse

    cen_y = float
    Coordinate y of the center of gravity of Hillas ellipse

    psi: float
    Angle between semimajor axis of the Hillas ellipse and the
    horizontal plane of the camera.

    Returns:
    --------
    float: source_x1

    float: source_x2
    
    """
    
    source_x1 = cen_x - disp*np.cos(psi)
    source_y1 = cen_y - disp*np.sin(psi)
   
    return source_x1,source_y1


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


def disp(source_pos, hillas):
    """
    Compute disp parameter

    Parameters
    ----------
    event: `ctapipe.io.container.MCEventContainer`
    tel: `ctapipe.instrument.TelescopeDescription`
    hillas: `ctapipe.io.container.HillasParametersContainer`

    Returns
    -------
    disp: float
    """
    disp = np.sqrt(((source_pos[0] - hillas.x) ** 2 + (source_pos[1] - hillas.y) ** 2).sum())
    return disp


def get_event_pos_in_sky(hillas, disp, tel, pointing_direction):
    side = 1  # TODO: method to guess side

    focal = tel.optics.equivalent_focal_length
    source_pos_in_camera = NominalFrame(array_direction=pointing_direction,
                                        pointing_direction=pointing_direction,
                                        x=(hillas.x + side * disp * np.cos(hillas.phi)) / focal * u.rad,
                                        y=(hillas.y + side * disp * np.sin(hillas.phi)) / focal * u.rad
                                        )

    horizon_frame = HorizonFrame(alt=pointing_direction.alt, az=pointing_direction.az)
    return source_pos_in_camera.transform_to(horizon_frame)



