#!/usr/bin/env python3

"""
Hillas parameters calculation of LST1 events from a simtelarray file.
Result is stored in a fits file. 
Running this script for several simtelarray files will concatenate events to the same fits file.

USAGE: python LST1_Hillas.py 'Particle' 'Simtelarray file' 'Store Img(true or false)' 

"""

import matplotlib.pylab as plt
import astropy.units as u
import numpy as np
import ctapipe
import os
import copy
import sys
from ctapipe.core import Container, Field, Map
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from ctapipe.image import hillas_parameters, hillas_parameters_2, tailcuts_clean
from ctapipe.io.eventsourcefactory import EventSourceFactory
from ctapipe.image.charge_extractors import LocalPeakIntegrator
from astropy.visualization import quantity_support
from astropy.table import vstack,Table
from astropy.io import fits

if __name__ == '__main__':
    
    #Some configuration variables
    ########################################################
    DATA_PATH="/home/queenmab/DATA/LST1/" # Path where Simtelarray files are stored
    TYPE=sys.argv[1] #Gamma, Proton, Electron (This are subfolders where simtelarray files of different type are stored)
    filename = sys.argv[2] #Name of the simtelarray file

    storeimg = sys.argv[3] #True for storing pixel information
    
    outfile = "/home/queenmab/DATA/LST1/Events/test.fits" #File where DL2 data will be stored 
    #######################################################
    
    #Cleaning levels:

    level1 = {'LSTCam' : 6.}
    level2 = level1.copy()
    # We use as second cleaning level just half of the first cleaning level
    for key in level2:
        level2[key] *= 0.5
    print(level2)
    
    source = EventSourceFactory.produce(input_url=DATA_PATH+TYPE+"/"+filename,allowed_tels={1}) #Open Simtelarray file 
    camtype = []   # one entry per image

    #Hillas Parameters
    width = np.array([])
    length = np.array([])
    phi = np.array([])
    psi = np.array([])
    r = np.array([])
    cen_x = np.array([])
    cen_y = np.array([])
    size = np.array([])
    
    #Event Parameters
    ObsID = np.array([])
    EvID = np.array([])
    
    #MC Parameters:
    mcEnergy = np.array([])
    mcAlt  = np.array([])
    mcAz = np.array([])
    mcCore_x = np.array([])
    mcCore_y = np.array([])
    mcHfirst = np.array([])
    mcType = np.array([])
    mcAlttel = np.array([])
    mcAztel = np.array([])
    GPStime = np.array([])

    fitsdata = np.array([])

    log10pixelHGsignal = {}
    survived = {}


    for key in level1:

        log10pixelHGsignal[key] = []
        survived[key] = []
    i=0
    for event in source:      
        if i%100==0:
            print("EVENT_ID: ", event.r0.event_id, "TELS: ",
                  event.r0.tels_with_data,
                  "MC Energy:", event.mc.energy )
        i=i+1
        ntels = len(event.r0.tels_with_data)

        '''
        if i > 100:   # for quick tests
            break
        '''
        for ii, tel_id in enumerate(event.r0.tels_with_data):
            
            geom = event.inst.subarray.tel[tel_id].camera #Camera geometry
                        
            data = event.r0.tel[tel_id].waveform
            ped = event.mc.tel[tel_id].pedestal
            # the pedestal is the average (for pedestal events) of the *sum* of all samples, from sim_telarray

            nsamples = data.shape[2]  # total number of samples
            pedcorrectedsamples = data - np.atleast_3d(ped)/nsamples    # Subtract pedestal baseline. atleast_3d converts 2D to 3D matrix

            integrator = LocalPeakIntegrator(None, None)
            integration, peakpos, window = integrator.extract_charge(pedcorrectedsamples) # these are 2D matrices num_gains * num_pixels

            chan = 0  # high gain used for now...
            signals = integration[chan].astype(float)

            dc2pe = event.mc.tel[tel_id].dc_to_pe   # numgains * numpixels
            signals *= dc2pe[chan]

            # Add all individual pixel signals to the numpy array of the corresponding camera inside the log10pixelsignal dictionary
            log10pixelHGsignal[str(geom)].extend(np.log10(signals))  # This seems to be faster like this, with normal python lists

            # Apply image cleaning
            cleanmask = tailcuts_clean(geom, signals, picture_thresh=level1[str(geom)],
                                       boundary_thresh=level2[str(geom)], keep_isolated_pixels=False, min_number_picture_neighbors=1)
            survived[str(geom)].extend(cleanmask)  # This seems to be faster like this, with normal python lists

            clean = signals.copy()
            clean[~cleanmask] = 0.0   # set to 0 pixels which did not survive cleaning
            if np.max(clean) < 1.e-6: # skip images with no pixels
                continue
            
            # Calculate image parameters
            hillas = hillas_parameters(geom, clean) # this one gives some warnings invalid value in sqrt            
            foclen = event.inst.subarray.tel[tel_id].optics.equivalent_focal_length

            w = np.rad2deg(np.arctan2(hillas.width,foclen));
            l = np.rad2deg(np.arctan2(hillas.length,foclen));

            if w >= 0:
                if fitsdata.size == 0:
                    fitsdata = clean
                else:
                    fitsdata = np.vstack([fitsdata,clean]) #Pixel content

                camtype.append(str(geom))
                width = np.append(width, w.value)
                length = np.append(length, l.value)
                phi = np.append(phi, hillas.phi)
                psi = np.append(psi, hillas.psi)
                r = np.append(r,hillas.r)
                cen_x = np.append(cen_x,hillas.cen_x)
                cen_y = np.append(cen_y,hillas.cen_y)
                size = np.append(size, hillas.size)
                

                #Store parameters from event and MC:
                ObsID = np.append(ObsID,event.r0.obs_id)
                EvID = np.append(EvID,event.r0.event_id)
            
                mcEnergy = np.append(mcEnergy,event.mc.energy)
                mcAlt = np.append(mcAlt,event.mc.alt)
                mcAz = np.append(mcAz,event.mc.az)
                mcCore_x = np.append(mcCore_x,event.mc.core_x)
                mcCore_y = np.append(mcCore_y,event.mc.core_y)
                mcHfirst = np.append(mcHfirst,event.mc.h_first_int)
                mcType = np.append(mcType,event.mc.shower_primary_id)
                mcAztel = np.append(mcAztel,event.mcheader.run_array_direction[0])
                mcAlttel = np.append(mcAlttel,event.mcheader.run_array_direction[1])
                
                GPStime = np.append(GPStime,event.trig.gps_time.value)

    #Store the output in an ntuple:
                
    output = {'camtype':camtype,'ObsID':ObsID,'EvID':EvID,'mcEnergy':mcEnergy,'mcAlt':mcAlt,'mcAz':mcAz, 'mcCore_x':mcCore_x,'mcCore_y':mcCore_y,'mcHfirst':mcHfirst,'mcType':mcType, 'GPStime':GPStime, 'width':width, 'length':length, 'phi':phi,'psi':psi,'r':r,'cen_x':cen_x,'cen_y':cen_y,'size':size,'mcAlttel':mcAlttel,'mcAztel':mcAztel}
    ntuple = Table(output)

    #If destination fitsfile doesn't exist, will create a new one with proper headers 
    if os.path.isfile(outfile)==False :
        #Convert Tables of data into HDUBinTables to write them into fits files
        pardata = ntuple.as_array()
        parheader = fits.Header()
        parheader.update(ntuple.meta)
        
        pixels = fits.ImageHDU(fitsdata) #Image with pixel content

        #Write the data in an HDUList for storing in a fitsfile
        hdr = fits.Header() #Example header, we can add more things to this header
        hdr['TEL'] = 'LST1'
        primary_hdu = fits.PrimaryHDU(header=hdr)
        hdul = fits.HDUList([primary_hdu])
        hdul.append(fits.BinTableHDU(data=pardata,header=parheader))
        if storeimg: hdul.append(pixels) 
        hdul.writeto(outfile)
    #If the destination fits file exists, will concatenate events:
    else:
        #If this is not the first data set, we must append the new data to the existing HDUBinTables and ImageHDU contained in the events.fits file.
        hdul=fits.open(outfile) #Open the existing file which contains two tables and 1 image
        #Get the already existing data:
        primary_hdu = hdul[0]
        data = Table.read(outfile,1)
        pixdata = hdul[2].data
        
        #Concatenate data
        data = vstack([data,ntuple])
        pixdata = np.vstack([pixdata,fitsdata])

        #Convert into HDU objects
        pardata = data.as_array()
        parheader = fits.Header()
        parheader.update(data.meta)
        pixhdu = fits.ImageHDU(pixdata)

        #Write the data in an HDUList for storing in a fitsfile
        
        hdul = fits.HDUList([primary_hdu])
        hdul.append(fits.BinTableHDU(data=pardata,header=parheader))
        hdul.append(pixhdu)
        
        hdul.writeto(outfile,overwrite=True)
                
        


    
    
