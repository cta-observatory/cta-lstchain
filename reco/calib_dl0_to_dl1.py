"""This is a module for extracting data from simtelarray files and 
calculate image parameters of the events: Hillas parameters, timing 
parameters. They can be stored in HDF5 file. The option of saving the 
full camera image is also available.

Usage:

"import calib_dl0_to_dl1"

"""
import numpy as np
from ctapipe.image import hillas_parameters, hillas_parameters_2, tailcuts_clean
from ctapipe.io.eventsourcefactory import EventSourceFactory
from ctapipe.image.charge_extractors import LocalPeakIntegrator
from ctapipe.utils import get_dataset
from ctapipe.image import timing_parameters as time
from ctapipe.instrument import OpticsDescription
import pandas as pd
import astropy.units as units
import h5py
import utils

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
    
def get_events(filename,storedata=False,
               concatenate=False,storeimg=False,outdir='./results/'):
    """
    Read a Simtelarray file, extract pixels charge, calculate image 
    parameters and timing parameters and store the result in an hdf5
    file. 
    
    Parameters:
    -----------
    filename: str
    Name of the simtelarray file.

    storedata: boolean
    True: store extracted data in a hdf5 file

    concatenate: boolean
    True: store the extracted data at the end of an existing file

    storeimg: boolean
    True: store also pixel data
    
    outdir: srt
    Output directory
    
    Returns:
    --------
    pandas DataFrame: output
    """
    #Particle type:
    
    particle_type = guess_type(filename)
    
    #Create data frame where DL2 data will be stored:

    features = ['ObsID','EvID','mcEnergy','mcAlt','mcAz','mcCore_x','mcCore_y',
                'mcHfirst','mcType','GPStime','width','length','w/l','phi',
                'psi','r','x','y','intensity','skewness','kurtosis','mcAlttel',
                'mcAztel','impact','mcXmax','time_gradient','intercept','SrcX',
                'SrcY','disp','hadroness']
    output = pd.DataFrame(columns=features)

    #Read LST1 events:
    source = EventSourceFactory.produce(
        input_url=filename, 
        allowed_tels={1}) #Open Simtelarray file

    #Cleaning levels:
        
    level1 = {'LSTCam' : 6.}
    level2 = level1.copy()
    # We use as second cleaning level just half of the first cleaning level
    
    for key in level2:
        level2[key] *= 0.5
    
    
    log10pixelHGsignal = {}
    survived = {}

    imagedata = np.array([])
    
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
            
            geom = event.inst.subarray.tel[tel_id].camera     #Camera geometry
            tel_coords = event.inst.subarray.tel_coords[
                event.inst.subarray.tel_indices[tel_id]
            ]
            
            data = event.r0.tel[tel_id].waveform
            
            ped = event.mc.tel[tel_id].pedestal    # the pedestal is the 
            #average (for pedestal events) of the *sum* of all samples,
            #from sim_telarray

            nsamples = data.shape[2]  # total number of samples
            
            # Subtract pedestal baseline. atleast_3d converts 2D to 3D matrix
            
            pedcorrectedsamples = data - np.atleast_3d(ped)/nsamples    
            
            integrator = LocalPeakIntegrator(None, None)
            integration, peakpos, window = integrator.extract_charge(
                pedcorrectedsamples) # these are 2D matrices num_gains * num_pixels
            
            chan = 0  # high gain used for now...
            signals = integration[chan].astype(float)

            dc2pe = event.mc.tel[tel_id].dc_to_pe   # numgains * numpixels
            signals *= dc2pe[chan]

            # Add all individual pixel signals to the numpy array of the
            # corresponding camera inside the log10pixelsignal dictionary
            
            log10pixelHGsignal[str(geom)].extend(np.log10(signals))  
            
            # Apply image cleaning
        
            cleanmask = tailcuts_clean(geom, signals, 
                                       picture_thresh=level1[str(geom)],
                                       boundary_thresh=level2[str(geom)], 
                                       keep_isolated_pixels=False, 
                                       min_number_picture_neighbors=1)
           
            survived[str(geom)].extend(cleanmask)  
           
            clean = signals.copy()
            clean[~cleanmask] = 0.0   # set to 0 pixels which did not 
            # survive cleaning
            
            if np.max(clean) < 1.e-6:  # skip images with no pixels
                continue
                
            # Calculate image parameters
        
            hillas = hillas_parameters(geom, clean)  
            foclen = event.inst.subarray.tel[tel_id].optics.equivalent_focal_length

            w = np.rad2deg(np.arctan2(hillas.width, foclen))
            l = np.rad2deg(np.arctan2(hillas.length, foclen))

            #Calculate Timing parameters
        
            peak_time = units.Quantity(peakpos[chan])*units.Unit("ns")
            timepars = time.timing_parameters(geom.pix_x,geom.pix_y,clean,peak_time,hillas.psi)
            
            if w >= 0:
                if storeimg==True:
                    if imagedata.size == 0:
                        imagedata = clean
                    else:
                        imagedata = np.vstack([imagedata,clean]) #Pixel content
                
                #Hillas parameters
                width = w.value
                length = l.value
                phi = hillas.phi.value
                psi = hillas.psi.value
                r = hillas.r.value
                x = hillas.x.value
                y = hillas.y.value
                intensity = np.log10(hillas.intensity)
                skewness =  hillas.skewness
                kurtosis = hillas.kurtosis
                
                #MC data:
                ObsID = event.r0.obs_id
                EvID = event.r0.event_id

                mcEnergy = np.log10(event.mc.energy.value*1e3) #Log10(Energy) in GeV
                mcAlt = event.mc.alt.value
                mcAz = event.mc.az.value
                mcCore_x = event.mc.core_x.value
                mcCore_y = event.mc.core_y.value
                mcHfirst = event.mc.h_first_int.value
                mcType = event.mc.shower_primary_id
                mcAztel = event.mcheader.run_array_direction[0].value
                mcAlttel = event.mcheader.run_array_direction[1].value
                mcXmax = event.mc.x_max.value
                GPStime = event.trig.gps_time.value

                #Calculate impact parameters

                impact = np.sqrt((
                    tel_coords.x.value-event.mc.core_x.value)**2
                    +(tel_coords.y.value-event.mc.core_y.value)**2)
                
                #Timing parameters

                time_gradient = timepars[0].value
                intercept = timepars[1].value

                #Calculate Disp and Source position in camera coordinates
                
                tel = OpticsDescription.from_name('LST') #Telescope description
                focal_length = tel.equivalent_focal_length.value
                sourcepos = utils.calc_CamSourcePos(mcAlt,mcAz,
                                                              mcAlttel,mcAztel,
                                                              focal_length) 
                SrcX = sourcepos[0]
                SrcY = sourcepos[1]
                disp = utils.calc_DISP(sourcepos[0],sourcepos[1],
                                                 x,y)
                
                hadroness = 0
                if particle_type=='proton':
                    hadroness = 1

                eventdf = pd.DataFrame([[ObsID,EvID,mcEnergy,mcAlt,mcAz,
                                         mcCore_x,mcCore_y,mcHfirst,mcType,
                                         GPStime,width,length,width/length,phi,
                                         psi,r,x,y,intensity,skewness,kurtosis,
                                         mcAlttel,mcAztel,impact,mcXmax,
                                         time_gradient,intercept,SrcX,SrcY,
                                         disp,hadroness]],
                                       columns=features)
                
                output = output.append(eventdf,
                                       ignore_index=True)

    outfile = outdir + particle_type + '_events.hdf5'
    
    if storedata==True:
        if (concatenate==False or 
            (concatenate==True and 
             np.DataSource().exists(outfile)==False)):
            output.to_hdf(outfile,
                          key=particle_type+"_events",mode="w")
            if storeimg==True:
                f = h5py.File(outfile,'r+')
                f.create_dataset('images',data=imagedata)
                f.close()
        else:
            if storeimg==True:
                f = h5py.File(outfile,'r')
                images = f['images']
                del f['images']
                images = np.vstack([images,imagedata])
                f.close()
                saved = pd.read_hdf(outfile,key=particle_type+'_events')
                output = saved.append(output,ignore_index=True)
                output.to_hdf(outfile,key=particle_type+"_events",mode="w")
                f = h5py.File(outfile,'r+')
                f.create_dataset('images',data=images)
                f.close()
            else:
                saved = pd.read_hdf(outfile,key=particle_type+'_events')
                output = saved.append(output,ignore_index=True)
                output.to_hdf(outfile,key=particle_type+"_events",mode="w")
    del source
    return output
