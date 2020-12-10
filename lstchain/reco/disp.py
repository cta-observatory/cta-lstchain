import numpy as np
from ..io import lstcontainers
from . import utils
import astropy.units as u

__all__ = [
    'disp',
    'miss',
    'disp_parameters',
    'disp_parameters_event',
    'disp_vector',
    'disp_to_pos'
    ]


def disp(cog_x, cog_y, src_x, src_y):
    """
    Compute the disp parameters

    Parameters
    ----------
    cog_x: `numpy.ndarray` or float
    cog_y: `numpy.ndarray` or float
    src_x: `numpy.ndarray` or float
    src_y: `numpy.ndarray` or float

    Returns
    -------
    (disp_dx, disp_dy, disp_norm, disp_angle, disp_sign):
        disp_dx: 'astropy.units.m`
        disp_dy: 'astropy.units.m`
        disp_norm: 'astropy.units.m`
        disp_angle: 'astropy.units.rad`
        disp_sign: `numpy.ndarray`
    """
    disp_dx = src_x - cog_x
    disp_dy = src_y - cog_y
    disp_norm = np.sqrt(disp_dx**2 + disp_dy**2)
    if hasattr(disp_dx, '__len__'):
        disp_angle = np.arctan(disp_dy / disp_dx)
        disp_angle[disp_dx == 0] = np.pi / 2. * np.sign(disp_dy[disp_dx == 0])
    else:
        if disp_dx == 0:
            disp_angle = np.pi/2. * np.sign(disp_dy)
        else:
            disp_angle = np.arctan(disp_dy/disp_dx)

    disp_sign = np.sign(disp_dx)

    return disp_dx, disp_dy, disp_norm, disp_angle, disp_sign


def miss(disp_dx, disp_dy, hillas_psi):
    """
    Compute miss

    Parameters
    ----------
    disp_dx: `numpy.ndarray` or float
    disp_dy: `numpy.ndarray` or float
    hillas_psi: `numpy.ndarray` or float

    Returns
    -------

    """
    return np.abs(np.sin(hillas_psi) * disp_dx - np.cos(hillas_psi)*disp_dy)


def disp_parameters(cog_x, cog_y, mc_alt, mc_az, mc_alt_tel, mc_az_tel, focal):
    """
    Compute disp parameters.

    Parameters
    ----------
    cog_x: `numpy.ndarray` or float
    cog_y: `numpy.ndarray` or float
    mc_alt: `numpy.ndarray` or float
    mc_az: `numpy.ndarray` or float
    mc_alt_tel: `numpy.ndarray` or float
    mc_az_tel: `numpy.ndarray` or float
    focal: `numpy.ndarray` or float

    Returns
    -------
    (disp_dx, disp_dy, disp_norm, disp_angle, disp_sign) : `numpy.ndarray` or float
    """
    source_pos_in_camera = utils.sky_to_camera(mc_alt, mc_az, focal, mc_alt_tel, mc_az_tel)
    return disp(cog_x, cog_y, source_pos_in_camera.x, source_pos_in_camera.y)



def disp_parameters_event(hillas_parameters, source_pos_x, source_pos_y):
    """
    Compute the disp_norm parameters from Hillas parameters in the event position in the camera frame
    Return a `DispContainer`

    Parameters
    ----------
    hillas_parameters: `ctapipe.containers.HillasParametersContainer`
    source_pos_x: `astropy.units.quantity.Quantity`
        X coordinate of the source (event) position in the camera frame
    source_pos_y: `astropy.units.quantity.Quantity`
        Y coordinate of the source (event) position in the camera frame

    Returns
    -------
    `lstchain.io.containers.DispContainer`
    """
    disp_container = lstcontainers.DispContainer()

    d = disp(hillas_parameters.x.to(u.m).value,
             hillas_parameters.y.to(u.m).value,
             source_pos_x.to(u.m).value,
             source_pos_y.to(u.m).value,
             )

    disp_container.dx = d[0] * u.m
    disp_container.dy = d[1] * u.m
    disp_container.norm = d[2] * u.m
    disp_container.angle = d[3] * u.rad
    disp_container.sign = d[4]
    disp_container.miss = miss(disp_container.dx.value,
                               disp_container.dy.value,
                               hillas_parameters.psi.to(u.rad).value) * u.m
    return disp_container



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
    return utils.polar_to_cartesian(disp_norm, disp_angle, disp_sign)


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
