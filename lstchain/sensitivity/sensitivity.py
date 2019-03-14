import numpy as np
import pandas as pd

from eventio.simtel.simtelfile import SimTelFile

def read_mc_par(source):
    emin, emax = source.mc_run_headers[0]['E_range']*1e3 #GeV
    spectral_index = source.mc_run_headers[0]['spectral_index']
    num_showers = source.mc_run_headers[0]['num_showers']
    num_use = source.mc_run_headers[0]['num_use']
    sim_ev= num_showers * num_use
    max_impact = source.mc_run_headers[0]['core_range'][1]*1e2 #cm
    area_sim = math.pi * math.pow(max_impact,2)
    cone = source.mc_run_headers[0]['viewcone'][1]    

    par = zip(emin, emax, spectral_index, num_showers, sim_ev, area_sim, cone)
    return par

def process(simtelfile, dl2_file):

    source = SimTelFile(simtelfile)
    mc_par = read_mc_par(source)
    events = pd.read_hdf(dl2_file)

    e_trig = 10**events.mc_energy
    n_trig = e_trig.shape[0]
    gammaness = events.gammaness
    theta2 = (events.src_x-events.src_x_rec)**2+(events.src_y-events.src_y)**2


def calculate_sensitivity(Ng, Nh, alpha):
    significance = (Ng)/np.sqrt(Nh * alpha)
    sensitivity = 5/significance * 100 # percentage of Crab
    
    return sensitivity


def sensitivity():
    simtelfile_gammas = "/home/queenmab/DATA/LST1/Gamma/gamma_20deg_0deg_run8___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel.gz"
    simtelfile_protons = "/home/queenmab/DATA/LST1/Proton/proton_20deg_0deg_run194___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel.gz"
    PATH_EVENTS = "../../cta-lstchain-extra/reco/sample_data/dl2/"
    dl2_file_g = PATH_EVENTS+"/reco_gammas.h5" ##Same events but with reconstructed 
    dl2_file_p = PATH_EVENTS+"/reco_protons.h5"

    gamma_result = process(simtelfile_gammas, dl2_file_g)
    proton_result = process(simtelfile_protons, dl2_file_p)


    obstime = 50 * 3600 # s (50 hours)

    final_gamma = np.ndarray(shape=(ebins,gammaness_bins,theta2_bins))
    final_hadrons = np.ndarray(shape=(ebins,gammaness_bins,theta2_bins))

    for i in range(0,eedges-1): # binning in energy
        e_w_binE = np.sum(e_w[(energies_g < E[i+1]) & (energies_g > E[i])])
        for g in range(0,gammaness_bins): # cut in gammaness
            Ngammas = []
            Nhadrons = []
            for t in range(0,theta2_bins): # cut in theta2
                e_trig_w_sum = np.sum(e_trig_w[(e_trig_g < E_trig[i+1]) & (e_trig_g > E_trig[i]) \
                                         & (gammaness_g > 0.1*g) & (theta2_g < 0.05*(t+1))])
            # Just considering all the hadrons give trigger...
                ep_w_sum = np.sum(ep_trig_w[(e_trig_p < E_trig[i+1]) & (e_trig_p > E_trig[i]) \
                                         & (gammaness_p > 0.1*g) & (theta2_p < 0.05*(t+1))])
            
                final_gamma[i][g][t] = e_trig_w_sum * obstime
                final_hadrons[i][g][t] = ep_w_sum * obstime

    
                
    sens = Calculate_sensititity(final_gamma, final_hadrons, 1)


    
