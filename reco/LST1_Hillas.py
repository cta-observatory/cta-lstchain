#!/usr/bin/env python3

"""
Hillas parameters calculation of LST1 events from a simtelarray file.
Result is stored in a fits file. 
Running this script for several simtelarray files will concatenate events to the same fits file.

USAGE: python LST1_Hillas.py 'Particle' 'Simtelarray file' 'Store Img(True or False)' 

"""

import numpy as np
import os
from ctapipe.image import hillas_parameters, hillas_parameters_2, tailcuts_clean
from ctapipe.io.eventsourcefactory import EventSourceFactory
from ctapipe.image.charge_extractors import LocalPeakIntegrator
import ctapipe.image.timing_parameters as time
from astropy.table import vstack,Table
from astropy.io import fits
import argparse
import h5py
from ctapipe.utils import get_dataset
import astropy.units as units

parser = argparse.ArgumentParser(description = "process CTA files.")

# Required argument
parser.add_argument('--filename', '-f', type=str,
                    dest='filename',
                    help='path to the file to process',
                    default=get_dataset('gamma_test.simtel.gz'))

# Optional arguments
parser.add_argument('--ptype', '-t', dest='particle_type', action='store',
                    default=None,
                    help='Particle type (gamma, proton, electron) - subfolders where simtelarray files of different type are stored)'
                         'Optional, if not passed, the type will be guessed from the filename'
                         'If not guessed, "unknown" type will be set'
                    )

parser.add_argument('--outdir', '-o', dest='outdir', action='store',
                    default='./results/',
                    help='Output directory to save fits file.'
                    )
parser.add_argument('--filetype','-ft', dest='filetype',action='store',
                    default='hdf5',type=str,
                    help='String. Type of output file: hdf5 or fits'
                    'Default=hdf5'
                    )
parser.add_argument('--storeimg', '-s', dest='storeimg', action='store',
                    default=False, type=bool,
                    help='Boolean. True for storing pixel information.'
                         'Default=False, any user input will be considered True'
                    )


args = parser.parse_args()


def guess_type(filename):
    """
    Guess the particle type from the filename

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


if __name__ == '__main__':

    #Some configuration variables
    ########################################################
    filename = args.filename
    particle_type = guess_type(filename) if args.particle_type is None else args.particle_type

    storeimg = args.storeimg

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    filetype = args.filetype
    outfile = args.outdir + '/' + particle_type + "_events."+filetype #File where DL2 data will be stored
    
    #######################################################

    #Cleaning levels:

    level1 = {'LSTCam' : 6.}
    level2 = level1.copy()
    # We use as second cleaning level just half of the first cleaning level
    for key in level2:
        level2[key] *= 0.5
    print(level2)

    source = EventSourceFactory.produce(input_url=filename, allowed_tels={1}) #Open Simtelarray file
    camtype = []   # one entry per image

    #Hillas Parameters
    width = np.array([])
    length = np.array([])
    phi = np.array([])
    psi = np.array([])
    r = np.array([])
    x = np.array([])
    y = np.array([])
    intensity = np.array([])

    skewness = np.array([])
    kurtosis = np.array([])
    
    #Event Parameters
    obs_id = np.array([])
    event_id = np.array([])

    #MC Parameters:
    mc_energy = np.array([])
    mc_alt  = np.array([])
    mc_az = np.array([])
    mc_core_x = np.array([])
    mc_core_y = np.array([])
    mc_h_first_int = np.array([])
    mc_type = np.array([])
    mc_alt_tel = np.array([])
    mc_az_tel = np.array([])
    mc_x_max = np.array([])
    gps_time = np.array([])

    impact = np.array([])
    
    fitsdata = np.array([])

    #Timing Parameters
    time_gradient = np.array([])
    intercept = np.array([])

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
            tel_coords = event.inst.subarray.tel_coords[event.inst.subarray.tel_indices[tel_id]]
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
            if np.max(clean) < 1.e-6:  # skip images with no pixels
                continue

            # Calculate image parameters
            hillas = hillas_parameters(geom, clean)  # this one gives some warnings invalid value in sqrt
            foclen = event.inst.subarray.tel[tel_id].optics.equivalent_focal_length

            w = np.rad2deg(np.arctan2(hillas.width, foclen))
            l = np.rad2deg(np.arctan2(hillas.length, foclen))

            #Calculate Timing parameters
        
            peak_time = units.Quantity(peakpos[chan])*units.Unit("ns")
            timepars = time.timing_parameters(geom.pix_x,geom.pix_y,clean,peak_time,hillas.psi)
            

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
                r = np.append(r, hillas.r)
                x = np.append(x, hillas.x)
                y = np.append(y, hillas.y)
                intensity = np.append(intensity, hillas.intensity)
                skewness =  np.append(skewness, hillas.skewness)
                kurtosis = np.append(kurtosis, hillas.kurtosis)
                
                #Store parameters from event and MC:
                obs_id = np.append(obs_id,event.r0.obs_id)
                event_id = np.append(event_id,event.r0.event_id)

                mc_energy = np.append(mc_energy,event.mc.energy)
                mc_alt = np.append(mc_alt,event.mc.alt)
                mc_az = np.append(mc_az,event.mc.az)
                mc_core_x = np.append(mc_core_x,event.mc.core_x)
                mc_core_y = np.append(mc_core_y,event.mc.core_y)
                mc_h_first_int = np.append(mc_h_first_int,event.mc.h_first_int)
                mc_type = np.append(mc_type,event.mc.shower_primary_id)
                mc_az_tel = np.append(mc_az_tel,event.mcheader.run_array_direction[0])
                mc_alt_tel = np.append(mc_alt_tel,event.mcheader.run_array_direction[1])
                mc_x_max = np.append(mc_x_max,event.mc.x_max)
                gps_time = np.append(gps_time,event.trig.gps_time.value)

                impact = np.append(impact,np.sqrt((tel_coords.x.value-event.mc.core_x.value)**2+(tel_coords.y.value-event.mc.core_y.value)**2))
                
                time_gradient = np.append(time_gradient,timepars[0])
                intercept = np.append(intercept,timepars[1])

    
    #Store the output in an ntuple:
              
    output = {'obs_id':obs_id,'event_id':event_id,'mc_energy':mc_energy,'mc_alt':mc_alt,'mc_az':mc_az, 'mc_core_x':mc_core_x,'mc_core_y':mc_core_y,'mc_h_first_int':mc_h_first_int,'mc_type':mc_type, 'gps_time':gps_time, 'width':width, 'length':length, 'phi':phi,'psi':psi,'r':r,'x':x,'y':y,'intensity':intensity,'skewness':skewness,'kurtosis':kurtosis,'mc_alt_tel':mc_alt_tel,'mc_az_tel':mc_az_tel,'impact':impact,'mc_x_max':mc_x_max,'time_gradient':time_gradient,'intercept':intercept}
    ntuple = Table(output)
    
    #If destination fitsfile doesn't exist, will create a new one with proper headers 
    if os.path.isfile(outfile)==False :
        if filetype=='fits':
            #Convert Tables of data into HDUBinTables to write them into fits files
            pardata = ntuple.as_array()
            parheader = fits.Header()
            parheader.update(ntuple.meta)

            if storeimg==True:
                pixels = fits.ImageHDU(fitsdata) #Image with pixel content
                
            #Write the data in an HDUList for storing in a fitsfile
            hdr = fits.Header() #Example header, we can add more things to this header
            hdr['TEL'] = 'LST1'
            primary_hdu = fits.PrimaryHDU(header=hdr)
            hdul = fits.HDUList([primary_hdu])
            hdul.append(fits.BinTableHDU(data=pardata,header=parheader))
            if storeimg==True:
                hdul.append(pixels) 
            hdul.writeto(outfile)
                    
        if filetype=='hdf5':
            f = h5py.File(outfile,'w')
            f.create_dataset(particle_type,data=ntuple.as_array())
            f.close()
    #If the destination fits file exists, will concatenate events:
    else:
        if filetype=='fits':
            #If this is not the first data set, we must append the new data to the existing HDUBinTables and ImageHDU contained in the events.fits file.
            hdul=fits.open(outfile) #Open the existing file which contains two tables and 1 image
            #Get the already existing data:
            primary_hdu = hdul[0]
            data = Table.read(outfile,1)
            if storeimg==True:
                pixdata = hdul[2].data
        
            #Concatenate data
            data = vstack([data,ntuple])
            if storeimg==True:
                pixdata = np.vstack([pixdata,fitsdata])

            #Convert into HDU objects
            pardata = data.as_array()
            parheader = fits.Header()
            parheader.update(data.meta)
            if storeimg==True:
                pixhdu = fits.ImageHDU(pixdata)

            #Write the data in an HDUList for storing in a fitsfile
        
            hdul = fits.HDUList([primary_hdu])
            hdul.append(fits.BinTableHDU(data=pardata,header=parheader))
            if storeimg==True:
                hdul.append(pixhdu)
        
            hdul.writeto(outfile,overwrite=True)
        
        if filetype=='hdf5':
            f = h5py.File(outfile,'r')
            key=list(f.keys())[0]
            data = np.array(f[key])
            data=Table(data)
            data = vstack([data,ntuple])
            f.close()
            f = h5py.File(outfile,'w')
            f.create_dataset(particle_type,data=data.as_array())
            f.close()
    
