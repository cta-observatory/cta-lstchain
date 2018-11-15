#!/usr/bin/env python3
'''
Module with auxiliar functions:
Transform AltAz coordinates into Camera coordinates (This should be implemented already in ctapipe but I haven't managed to find how to do it)
Calculate source position from disp_ distance.
Calculate disp_ distance from source position.

Usage:

"import transformations"

'''

import numpy as np
import ctapipe.coordinates as c
import astropy.units as u

def alt_to_theta(alt):
    """
    transforms altitude (angle from the horizon upwards) to theta (angle from z-axis)                                                         
    for simtel array coordinate systems                                  
    """
    return (90 * u.deg - alt).to(alt.unit)


def az_to_phi(az):
    """
    transforms azimuth (angle from north towards east)                
    to phi (angle from x-axis towards y-axis)                            
    for simtel array coordinate systems                                  
    """
    return -az

        
def cal_cam_source_pos(mc_alt,mc_az,mc_alt_tel,mc_az_tel,focal_length):
    """
    Transform Alt-Az source position into Camera(x,y) coordinates source position. 

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
    #    for (alttel,aztel) in zip(mc_alt_tel,mc_az_tel):
        
    alttel = mc_alt_tel
    aztel = mc_az_tel
    mat_Y = np.array([[np.cos(alttel),0,np.sin(alttel)],
                      [0,1,0], 
                      [-np.sin(alttel),0,np.cos(alttel)]]).T
        
        
    mat_Z = np.array([[np.cos(aztel),-np.sin(aztel),0],
                      [np.sin(aztel),np.cos(aztel),0],
                      [0,0,1]]).T
    
        
    #rot_Matrix = np.append(rot_Matrix,[np.matmul(mat_Y,mat_Z)],axis=0)
    rot_Matrix = np.matmul(mat_Y,mat_Z)
    
    res = np.einsum("...ji,...i",rot_Matrix,source)
    res = res.T
    
    source_x = -focal_length*res[0]/res[2]
    source_y = -focal_length*res[1]/res[2]
    return source_x, source_y

def calc_disp(source_x,source_y,cen_x,cen_y):
    """
    Calculates "disp_" distance from source position in camera coordinates
    """
    disp = np.sqrt((source_x-cen_x)**2+(source_y-cen_y)**2)
    return disp

def disp__to_Pos(disp_,cen_x,cen_y,psi):
    """
    Calculates source position in camera coordinates(x,y) from "disp_" distance.
    For now, it only works for POINT GAMMAS, it doesn't take into account the duplicity of the 
    disp method. 
    """
    
    source_x1 = cen_x - disp_*np.cos(psi)
    source_y1 = cen_y - disp_*np.sin(psi)
   
    return source_x1,source_y1
        
       
