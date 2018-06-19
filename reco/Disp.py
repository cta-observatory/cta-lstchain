#!/usr/bin/env python3
'''
Module for calculating Source position in camera coordinates, and Disp distance.
Usage:

import Disp

'''

import numpy as np
import ctapipe.coordinates as c
import astropy.units as u

def calc_CamSourcePos(mcAlt,mcAz,mcAlttel,mcAztel,focal_length):
    mcAlt = c.alt_to_theta(mcAlt*u.rad).value
    mcAz = c.az_to_phi(mcAz*u.rad).value
    mcAlttel = c.alt_to_theta(mcAlttel*u.rad).value
    mcAztel = c.az_to_phi(mcAztel*u.rad).value

    #Sines and cosines of direction angles
    cp = np.cos(mcAz)
    sp = np.sin(mcAz)
    ct = np.cos(mcAlt)
    st = np.sin(mcAlt)
    
     #Shower direction coordinates

    sourcex = st*cp
    sourcey = st*sp
    sourcez = ct

    #print(sourcex)

    source = np.array([sourcex,sourcey,sourcez])
    source=source.T
    
    #Rotation matrices towars the camera frame
    
    rot_Matrix = np.empty((0,3,3))
    for (alttel,aztel) in zip(mcAlttel,mcAztel):
        
        mat_Y = np.array([[np.cos(alttel),0,np.sin(alttel)],
                        [0,1,0], 
                        [-np.sin(alttel),0,np.cos(alttel)]]).T
        
        
        mat_Z = np.array([[np.cos(aztel),-np.sin(aztel),0],
                        [np.sin(aztel),np.cos(aztel),0],
                        [0,0,1]]).T
    
        
        rot_Matrix = np.append(rot_Matrix,[np.matmul(mat_Y,mat_Z)],axis=0)
    
    res = np.einsum("...ij,...i",rot_Matrix,source)
    res = res.T
    
    Source_X = -focal_length*res[0]/res[2]
    Source_Y = -focal_length*res[1]/res[2]
    return Source_X, Source_Y

def calc_DISP(Source_X,Source_Y,cen_x,cen_y):
    disp = np.sqrt((Source_X-cen_x)**2+(Source_Y-cen_y)**2)
    return disp
