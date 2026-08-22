import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import animation
import os

import glob
import tables

#from lstchain.reco.utils import get_effective_time, add_delta_t_key
from ctapipe.instrument import SubarrayDescription
from ctapipe.visualization import CameraDisplay
from ctapipe.coordinates import EngineeringCameraFrame
#from ctapipe.instrument import CameraGeometry
from ctapipe.io import EventSource


#################################################################################
# function for printing a set of images and save them into a directory
#################################################################################


def plot(      array_ids,                     #array with the ID's of the events that we want to represent
         
               config,                        #configuration data

               representation='charge',       #type of graphic representation 'charge', 'time' or 'both'

               plot_direction=True,           #to represent the reconstructed direction, ellipse, and source

               gamma_lim=0.5,                 #gammanness limit to represent the reconstructed parameters 

               colormap_scale=0.7,            #change the maximum limit of the colormap [0-1]

               save=True,                     #saving the image into a folder

               file_type='.pdf',              #type of archive to save the image '.pdf', '.png' etc

               plot_image=True,               #for plotting or not the image at the console

               folder_name='event_plots'      #name of the folder for saving the plots
        
        ):

    #################################################
    
    #definition of the directories 
    number_subruns=config[0]
    run_number=config[1]
    subrun_events=config[2]
    dl2_directory=config[3]
    dl1_parameters=config[4]
    dl2_parameters=config[5]
    dl1_directory=config[6]
    
    #################################################

    #reordering events and identifying the corresponding subrun in order to optimize the plotting

    #if we input only one event in format 'int' we convert-it to a array
    if type(array_ids) is int:
        array_ids=[array_ids]

    array_ids.sort()
    #we separate the events in each subrun
    SubrunsList,SubrunIndex=[],[]
    for i in range(number_subruns+1):
        SubrunsList.append([])
        SubrunIndex.append(i)

    for i in range(len(array_ids)):
        for k in range(number_subruns+1):
            if  array_ids[i]>=(k*subrun_events) and array_ids[i]<((k+1)*subrun_events):
                SubrunsList[k].append(array_ids[i])

    LenSubrun=[]       
    for i in range(number_subruns+1):
        LenSubrun.append(len(SubrunsList[i]))

    #we only iterate the subruns that have some events
    SubrunsList_,SubrunIndex_=[],[]
    for i in range(len(SubrunsList)):
        if LenSubrun[i]!=0:
            SubrunsList_.append(SubrunsList[i])
            SubrunIndex_.append(SubrunIndex[i])

    #################################################

    #importing dl2 data
    
    df=pd.read_hdf(dl2_directory, dl2_parameters)  
    df=df.set_index('event_id')

    #################################################

    #definition of functions for plotting

    #############################
    #plot a basic charge map#####
    def basic_plot(N,represent='charge'):

        #titles, we print the value of gammaness and energy if exist dl2
        if (df.index == N).any()==True:
            titles='$\\gamma=$'+ str(round((df.at[N,'gammaness']),2))+',   $E=$'+str(round((df.at[N,'reco_energy']),2))+'TeV'
        else:
            titles='Event ID = '+str(N)

        if represent=='time':
            charge_map = times_data[N-subrun_events*SubrunIndex_[ii]-1]
            camdisplay = CameraDisplay(camera_geom.transform_to(EngineeringCameraFrame()),
                                       norm='lin', title='Time  (ns)' ,image=charge_map,cmap='Reds',
                                        show_frame=False)
        else:
            charge_map = charges_data[N-subrun_events*SubrunIndex_[ii]-1]
            camdisplay = CameraDisplay(camera_geom.transform_to(EngineeringCameraFrame()),
                                       norm='lin', title=titles ,image=charge_map,cmap='plasma',
                                        show_frame=False)     

        camdisplay.add_colorbar()
        camdisplay.set_limits_percent(colormap_scale*100)

    ##############################
    #complet plot of an event#####
    def complet_plot(N,represent='charge'):

        print('Plotting Event ID ='+str(N))

        fig, ax=plt.subplots()

        if only_dl1==False:

            basic_plot(N,represent)

            if plot_direction==True:
                #Source reconstruction
                plt.plot(-df.at[N,'reco_src_y'],-df.at[N,'reco_src_x'],'*',color='darkgrey',
                         label='Reconstructed source',markersize=17,alpha=0.9)

                plt.autoscale(False)
                #ellipse and mass center
                plt.plot(-df.at[N,'y'],-df.at[N,'x'],'.',color='w')
                ellipse = Ellipse(xy=(-df.at[N,'y'], -df.at[N,'x']), width=df.at[N,'width'], 
                                  height=df.at[N,'length'],angle=-np.rad2deg(df.at[N,'psi']), 
                                  edgecolor='w', fc='None', lw=2)
                ax.add_patch(ellipse)
                #print a line of direction
                slope=np.tan(-df.at[N,'psi']+np.pi/2)
                x0=-df.at[N,'y']
                y0=-df.at[N,'x']
                plt.plot([(3-y0+slope*x0)/slope,(-3-y0+slope*x0)/slope],[3,-3],'--',color='w')

                plt.legend(loc='best')

        elif only_dl1==True:
            basic_plot(N,represent)   

        #saving the images in a folder
        if save==True:
            if not os.path.exists(folder_name+'_Run'+str(run_number)):
                os.makedirs(folder_name+'_Run'+str(run_number))

            fig.savefig(folder_name+'_Run'+str(run_number)+'/event_'+str(N).zfill(7)+
                        '_'+representation+file_type, dpi=300, bbox_inches='tight')

        #if we only want to download the image we dont show it (is faster)
        if plot_image==False:
            plt.close()
        else:
            plt.show()

    ###########################################################################
    #plot a representation of both charges and times one next to the other#####
    def complet_plot_double(N):

        print('Plotting Event ID ='+str(N))
        fig, ax=plt.subplots(figsize=(17,7))

        #first image of charges
        ax1=plt.subplot(1,2,1)

        if only_dl1==False:

            basic_plot(N)

            if plot_direction==True:
                #Source reconstruction
                plt.plot(-df.at[N,'reco_src_y'],-df.at[N,'reco_src_x'],'*',
                         color='darkgrey',label='Reconstructed source',markersize=17,alpha=0.9)
                plt.autoscale(False)
                #ellipse and center of mass
                plt.plot(-df.at[N,'y'],-df.at[N,'x'],'.',color='w')
                ellipse = Ellipse(xy=(-df.at[N,'y'], -df.at[N,'x']), width=df.at[N,'width'], 
                                  height=df.at[N,'length'],angle=-np.rad2deg(df.at[N,'psi']), 
                                  edgecolor='w', fc='None', lw=2)
                ax1.add_patch(ellipse)
                #print a line of direction 
                slope=np.tan(-df.at[N,'psi']+np.pi/2)
                x0=-df.at[N,'y']
                y0=-df.at[N,'x']
                plt.plot([(3-y0+slope*x0)/slope,(-3-y0+slope*x0)/slope],[3,-3],'--',color='w')

                plt.legend(loc='best')

        elif only_dl1==True:
            basic_plot(N)        

        #second image of times
        plt.subplot(1,2,2)

        basic_plot(N,'time')

        #saving the images in a folder
        if save==True:
            if not os.path.exists(folder_name+'_Run'+str(run_number)):
                os.makedirs(folder_name+'_Run'+str(run_number))

            fig.savefig(folder_name+'_Run'+str(run_number)+'/event_'+str(N).zfill(7)+
                        '_'+representation+file_type, dpi=300, bbox_inches='tight')

        #if we only want to download the image we dont show 
        if plot_image==False:
            plt.close()
        else:
            plt.show()

    #################################################

    #plotting

    #plot parameters in a context to not affect the before defined parameters
    with plt.rc_context(rc={'figure.figsize':(10,9),
                            'font.size':17,
#                            'mathtext.fontset':'custom',
                            'mathtext.rm':'Bitstream Vera Sans',
                            'mathtext.it':'Bitstream Vera Sans:italic',
                            'mathtext.bf':'Bitstream Vera Sans:bold',
                            'mathtext.fontset':'stix',
                            'font.family':'STIXGeneral',
                            'xtick.direction':'out',
                            'ytick.direction':'out',
                            'xtick.major.size':8,
                            'xtick.major.width':2,
                            'xtick.minor.size':5,
                            'xtick.minor.width':1,
                            'ytick.major.size':8,
                            'ytick.major.width':2,
                            'ytick.minor.size':5,
                            'ytick.minor.width':1,
                            'xtick.top':False,
                            'ytick.right':False,
                            'xtick.minor.visible':False,
                            'ytick.minor.visible':False}):

        #we iterate the process subrun to subrun
        for ii in range(len(SubrunIndex_)):  

            #importing the DL1 data of corresponding subrun

            data_files = dl1_directory+str(SubrunIndex_[ii]).zfill(4)+".h5"

            dummy = []
            data_files = glob.glob(data_files)
            data_files.sort()
            for data_file in data_files:
                dfDL1 = pd.read_hdf(data_file, dl1_parameters)
                dummy.append(dfDL1)
            # data_parameters = pd.concat(dummy, ignore_index=True)
            subarray_info = SubarrayDescription.from_hdf(data_files[0])
            # focal_length = subarray_info.tel[1].optics.equivalent_focal_length
            camera_geom = subarray_info.tel[1].camera.geometry
            dummy1 = []
            dummy2 = []
            for data_file in data_files:
                data = tables.open_file(data_file)
                dummy1.append(data.root.dl1.event.telescope.image.LST_LSTCam.col('image'))
                dummy2.append(data.root.dl1.event.telescope.image.LST_LSTCam.col('peak_time'))
            charges_data = np.concatenate(dummy1) 
            times_data = np.concatenate(dummy2)


            #we plot each event jj for a determined subrun ii

            for jj in SubrunsList_[ii]:

                #for each event first we see if exist data from dl2
                only_dl1=True
                if plot_direction==True:
                    if (df.index == jj).any()==True:
                        if df.at[jj,'gammaness']>gamma_lim:
                            only_dl1=False

                #depending the representation choosen we use a method of plotting
                if representation=='both':
                    complet_plot_double(jj)

                elif representation=='time':
                    complet_plot(jj,'time')
                else:
                    complet_plot(jj,'charge')


            #################################################



#################################################################################
# function for printing a set of animations and save them into a directory
#################################################################################


def animate(   array_ids,                     #array with the ID's of the events that we want to represent
            
               config,                        #configuration data

               plot_direction=True,           #to represent the reconstructed direction, ellipse, and source

               gamma_lim=0.5,                 #gammanness limit to represent the reconstructed parameters

               colormap_scale=0.7,            #change the maximum limit of the colormap [0-1]

               file_type='.gif',              #type of archive to save the image '.gif' (for '.mp4' ffpmeg needed)

               fps=20,                        #frames per second for the animation

               folder_name='event_animations' #name of the folder for saving the plots

               ):

    #################################################
    
    #definition of the directories 
    number_subruns=config[0]
    run_number=config[1]
    subrun_events=config[2]
    dl2_directory=config[3]
    # dl1_parameters=config[4]
    dl2_parameters=config[5]
    # dl1_directory=config[6]
    R0_directory=config[7]
    # calibration_directory = config[8]
    # drs4_pedestal_path = config[9]
    # calib_path = config[10]
    # time_calib_path = config[11]
    configuration=config[12]
    
    #################################################

    #reordering events and identifying the corresponding subrun in order to optimize the plotting

    #if we input only one event in format 'int' we convert-it to a array
    if type(array_ids) is int:
        array_ids=[array_ids]

    array_ids.sort()
    #we separate the events in each subrun
    SubrunsList=[]
    SubrunsList_translate=[]
    SubrunIndex=[]
    for i in range(number_subruns+1):
        SubrunsList.append([])
        SubrunsList_translate.append([])
        SubrunIndex.append(i)

    for i in range(len(array_ids)):
        for k in range(number_subruns+1):
            if  array_ids[i]>=(k*subrun_events) and array_ids[i]<((k+1)*subrun_events):
                SubrunsList[k].append(array_ids[i])

    LenSubrun=[]       
    for i in range(number_subruns+1):
        LenSubrun.append(len(SubrunsList[i]))

    #we oly iterate the subruns with events
    SubrunsList_,SubrunIndex_,SubrunsList_translate_=[],[],[]
    for i in range(len(SubrunsList)):
        if LenSubrun[i]!=0:
            SubrunsList_.append(SubrunsList[i])
            SubrunsList_translate_.append(SubrunsList_translate[i])
            SubrunIndex_.append(SubrunIndex[i])

    #translating event id's into positions in a list of each subrun
    for i in range(len(SubrunsList_)):
        for j in range(len(SubrunsList_[i])):
            if j==0:
                SubrunsList_translate_[i].append(SubrunsList_[i][j]-SubrunIndex_[i]*subrun_events)
            else:
                SubrunsList_translate_[i].append(SubrunsList_[i][j]-SubrunsList_[i][j-1])

    #estimation of time of the process
    total_iterations=0
    total_images=0
    for i in range(len(SubrunsList_translate_)):
        for j in range(len(SubrunsList_translate_[i])):
            total_iterations=total_iterations+SubrunsList_translate_[i][j]
            total_images=total_images+1
    time_est=(33*total_iterations/1000+total_images*12)/60
    print('\n Estimated time = '+str(round((time_est),2))+' (min) \n')        


    #################################################

    #import dl2 data if needed

    if plot_direction==True:
        df=pd.read_hdf(dl2_directory, dl2_parameters)  
        df=df.set_index('event_id')

    #################################################

    #definition of function to animate outside of the loop

    def animate(i):

        camdisplay.image=ev.r1.tel[1].waveform[:,i]
        plt.title('$t=$'+str(i).zfill(2)+' (ns)')

        return fig,

    #################################################

    #plotting

    #plot parameters in a context to not affect the before defined parameters
    with plt.rc_context(rc={'figure.figsize':(10,9),
                            'font.size':17,
                            # 'mathtext.fontset':'custom',
                            'mathtext.rm':'Bitstream Vera Sans',
                            'mathtext.it':'Bitstream Vera Sans:italic',
                            'mathtext.bf':'Bitstream Vera Sans:bold',
                            'mathtext.fontset':'stix',
                            'font.family':'STIXGeneral',
                            'xtick.direction':'out',
                            'ytick.direction':'out',
                            'xtick.major.size':8,
                            'xtick.major.width':2,
                            'xtick.minor.size':5,
                            'xtick.minor.width':1,
                            'ytick.major.size':8,
                            'ytick.major.width':2,
                            'ytick.minor.size':5,
                            'ytick.minor.width':1,
                            'xtick.top':False,
                            'ytick.right':False,
                            'xtick.minor.visible':False,
                            'ytick.minor.visible':False}):

        #we iterate the process subrun to subrun
        for ii in range(len(SubrunIndex_)):    

            #importing the DL1 data of corresponding subrun
            R0_file = R0_directory+str(SubrunIndex_[ii]).zfill(4)+".fits.fz"


            #"Event source", to read in the R0 (=raw) events and calibrate them into R1 (=calibrated waveforms)
            source = EventSource(input_url=R0_file, config=configuration, max_events=5000)

            #################################################

            #we plot each event jj for a determined subrun ii

            for jj in SubrunsList_translate_[ii]: 

                #for going to a determined event ID we need to pass trough all before id's (don't know a faster way to do it) 
                #aproximately pass trough 1000 events in 30 seconds
                for j in range(jj):
                    for i, ev in enumerate(source): 
                        break

                # event_id, the same meaning as in DL1 and DL2 files:
                index=ev.index.event_id

                camgeom = source.subarray.tel[1].camera.geometry

                #we see if we have data in dl2 of the corresponding event
                only_dl1=True
                if plot_direction==True:
                    if (df.index == index).any()==True:
                        if df.at[index,'gammaness']>gamma_lim:
                            only_dl1=False

                #find maximum value
                max_=[]
                for i in range(36):
                        max_.append(max(ev.r1.tel[1].waveform[:,i]))
                maximum=max(max_)

                #################################################

                #plotting

                fig, ax=plt.subplots(figsize=(13,11))

                camdisplay =CameraDisplay(camgeom.transform_to(EngineeringCameraFrame()),ax=ax,
                                            image=ev.r1.tel[1].waveform[:,0],
                                            show_frame=False,cmap='plasma')
                #setting the limits
                camdisplay.add_colorbar()
                camdisplay.set_limits_minmax(0, maximum*colormap_scale)

                anim = animation.FuncAnimation(fig, animate,frames=36, interval=22, blit=True)

                if (plot_direction==True) and (only_dl1==False):
                    #Source reconstruction
                    plt.plot(-df.at[index,'reco_src_y'],-df.at[index,'reco_src_x'],'*',color='darkgrey',
                             label='Reconstructed source',markersize=17 ,alpha=0.9)
                    #print a line of direction
                    plt.autoscale(False)
                    slope=np.tan(-df.at[index,'psi']+np.pi/2)
                    x0=-df.at[index,'y']
                    y0=-df.at[index,'x']
                    plt.plot([(3-y0+slope*x0)/slope,(-3-y0+slope*x0)/slope],[3,-3],'--',color='w',alpha=0.8)

                    plt.legend(loc='best')
                  
                fig.subplots_adjust(left=0.09, bottom=0.09, right=1.05, top=0.95, wspace=None, hspace=None)
                
                #saving the animations in a folder
                if not os.path.exists(folder_name+'_Run'+str(run_number)):
                    os.makedirs(folder_name+'_Run'+str(run_number))

                anim.save(folder_name+'_Run'+str(run_number)+'/event_'+str(index).zfill(7)
                            +file_type, fps=fps, extra_args=['-vcodec', 'libx264'])

                plt.close()

                #################################################


#################################################################################
# function for searching in the events of the run
#################################################################################


def search( config,                         #configuration data
    
            sort=False,                     #if we want to sort the list in terms of some variable for example ='gammaness'

            inflim_index=False,             #index of the event ID
            suplim_index=False,

            inflim_gammaness=False,         #gammaness
            suplim_gammaness=False,

            inflim_intensity=False,         #intensity
            suplim_intensity=False,

            inflim_proportion=False,        #proportion between length and width
            suplim_proportion=False,

            inflim_length=False,            #length
            suplim_length=False,

            inflim_width=False,             #width
            suplim_width=False,

            inflim_time_gradient=False,     #time gradient
            suplim_time_gradient=False,

            inflim_reco_energy=False,       #reco_energy
            suplim_reco_energy=False,

            #defoult defined parameters
            #gamma like intense events
            gamma_like=False,
            #muon like events
            muon_like=False,
          ):

    #################################################
    
    #definition of the directories 

    dl2_directory=config[3]
    dl2_parameters=config[5]
    
    #################################################
    
    
    df = pd.read_hdf(dl2_directory,dl2_parameters)

    #calculus the proportion of width/lenght
    if (inflim_proportion!=False) or (suplim_proportion!=False) or (sort=='proportion') or (muon_like!=False):
        df['proportion']=df['width']/df['length']

    #calculus of the absolute value of time gradient
    if (inflim_time_gradient!=False) or (suplim_time_gradient!=False) or (sort=='time_gradient'):
        df['time_gradient']=abs(df['time_gradient'])

    df=df.set_index('event_id')

    ############################

    #predefined filters

    #gammalike events
    if gamma_like==True:
        sort='gammaness'

        # inflim_gammanness=0.83
        inflim_intensity=1200,

    #muonlike events
    if muon_like==True:
        sort='proportion'

        inflim_intensity=300,
        inflim_proportion=0.78
        inflim_width=0.6  

    ############################

    #filters
    #we doesnt filter the parameters that are not defined to optimize

    #restrict only cosmic events
    df=df[df['event_type']==32]

    #number of event
    df=df[df.index > [1]] 
    if inflim_index!=False:
        df=df[df.index >= [inflim_index]] 
    if suplim_index!=False:
        df=df[df.index <= [suplim_index]] 

    #intensity
    if inflim_intensity!=False:
        df=df[df['intensity']>=inflim_intensity]  
    if suplim_intensity!=False:
        df=df[df['intensity']<=suplim_intensity]

    #reconstructed energy
    if inflim_reco_energy!=False:
        df=df[df['reco_energy']>=inflim_reco_energy]
    if suplim_reco_energy!=False:
        df=df[df['reco_energy']<=suplim_reco_energy] 

    #gammaness
    if inflim_gammaness!=False:
        df=df[df['gammaness']>=inflim_gammaness]  
    if suplim_gammaness!=False:
        df=df[df['gammaness']<=suplim_gammaness]   

    #width
    if inflim_width!=False:
        df=df[df['width']>=inflim_width] 
    if suplim_width!=False:
        df=df[df['width']<=suplim_width]  

    #length
    if inflim_length!=False:
        df=df[df['length']>=inflim_length]  
    if suplim_length!=False:
        df=df[df['length']<=suplim_length] 

    #proportion
    if inflim_proportion!=False:
        df=df[df['proportion']>=inflim_proportion]  
    if suplim_proportion!=False:
        df=df[df['proportion']<=suplim_proportion] 

    #time_gradient
    if inflim_time_gradient!=False:
        df=df[df['time_gradient']>=inflim_time_gradient]  
    if suplim_time_gradient!=False:
        df=df[df['time_gradient']<=suplim_time_gradient] 

    #also we can order from higher to lower for some variable
    if sort!=False:
        df=df.sort_values(sort,ascending=False)

    ############################

    #return an array with the event id's

    return list(df.index.values)   

    ############################
