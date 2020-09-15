#!/usr/bin/env python3

"""
Script to compute the LST sensitivity using MC.

Inputs are DL1/DL2 gamma and proton files

Usage: 

$> python lstchain_mc_sensitivity.py
--gd2-cuts dl2_gammas_cuts.h5  
--pd2-cuts dl2_protons_cuts.h5 
--gd2-sens dl2_gammas_sensitivity.h5 
--pd2-sens dl2_protons_sensitivity.h5
--o /output/path

"""


from lstchain.mc.sensitivity import sensitivity, find_best_cuts_sensitivity
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table
import numpy as np
import pandas as pd
import argparse
import ctaplot
from lstchain.visualization import plot_dl2
from lstchain.reco import utils
import seaborn as sns
from lstchain.io import read_simu_info_merged_hdf5
from lstchain.io.io import dl2_params_lstcam_key
from lstchain.spectra.crab import crab_hegra
from lstchain.mc import plot_utils

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=RuntimeWarning)

ctaplot.set_style()


parser = argparse.ArgumentParser(description="Compute MC sensitivity curve.")

parser.add_argument('--input-file-gamma-dl2', '--gdl2', type = str,
                    dest = 'dl2_file_g',
                    help = 'path to reconstructed gammas dl2 file')
parser.add_argument('--input-file-proton-dl2', '--pdl2', type = str,
                    dest = 'dl2_file_p',
                    help = 'path to reconstructed protons dl2 file')
parser.add_argument('--output_path', '--o', type = str,
                    dest = 'output_path',
                    help = 'path where to save plot images')

args = parser.parse_args()


def main():
    ntelescopes_gamma = 4
    ntelescopes_protons = 4
    n_bins_energy = 20  #  Number of energy bins
    n_bins_gammaness = 10  #  Number of gammaness bins
    n_bins_theta2 = 10  #  Number of theta2 bins
    obstime = 50 * 3600 * u.s
    noff = 5
    fraction_of_events_for_cuts = 0.5 # Fraction of the total number
    #of events to be used to calculate the best sensitivity cuts

    #Divide the event set in two:
    #First half for calculating the best sensitivity cuts
    #Second half for calcularing the sensitivity

    df_gammas=pd.read_hdf(args.dl2_file_g,
                          key=dl2_params_lstcam_key)
    df_protons=pd.read_hdf(args.dl2_file_p,
                          key=dl2_params_lstcam_key)

    #Must be a better way but I can't check the internet...
    gammas_array=df_gammas.to_numpy()
    protons_array=df_protons.to_numpy()

    half_size_gammas=round(gammas_array.shape[0]*fraction_of_events_for_cuts)
    half_size_protons=round(protons_array.shape[0]*fraction_of_events_for_cuts)
    
    gamma_events_for_cuts=gammas_array[:half_size_gammas]
    gamma_events_for_sens=gammas_array[half_size_gammas:]

    proton_events_for_cuts=protons_array[:half_size_protons]
    proton_events_for_sens=protons_array[half_size_protons:]

    #Check that the sizes are correct
    if gamma_events_for_cuts.shape[0]+gamma_events_for_sens.shape[0]!=gammas_array.shape[0]:
        print("Oops! The total is not the sum of the halves!")
        return
    
    if proton_events_for_cuts.shape[0]+proton_events_for_sens.shape[0]!=protons_array.shape[0]:
        print("Oops! The total is not the sum of the halves!")
        return

    #Create dataframes with the new two data sets
    df_gamma_events_for_cuts=pd.DataFrame(
        data=gamma_events_for_cuts,
        columns=df_gammas.keys())

    df_gamma_events_for_sens=pd.DataFrame(
        data=gamma_events_for_sens,
        columns=df_gammas.keys())
    
    df_proton_events_for_cuts=pd.DataFrame(
        data=proton_events_for_cuts,
        columns=df_protons.keys())

    df_proton_events_for_sens=pd.DataFrame(
        data=proton_events_for_sens,
        columns=df_protons.keys())
    
    ######################################################

    
    # Finds the best cuts for the computation of the sensitivity
    energy, best_sens, result, units, gcut, tcut = find_best_cuts_sensitivity(args.dl2_file_g,
                                                                              args.dl2_file_p,
                                                                              df_gamma_events_for_cuts,
                                                                              df_proton_events_for_cuts,
                                                                              ntelescopes_gamma,
                                                                              ntelescopes_protons,
                                                                              n_bins_energy,
                                                                              n_bins_gammaness,
                                                                              n_bins_theta2,
                                                                              noff,
                                                                              fraction_of_events_for_cuts,
                                                                              obstime)
                                                                              
                                                                              
    #For testing using fixed cuts
    #gcut = np.ones(n_bins_energy) * 0.8 
    #tcut = np.ones(n_bins_energy) * 0.01
    
    print("\nApplying optimal gammaness cuts:", gcut)
    print("Applying optimal theta2 cuts: {} \n".format(tcut))


    # Computes the sensitivity
    energy, best_sens, result, units, dl2 = sensitivity(args.dl2_file_g, 
                                                        args.dl2_file_p,
                                                        df_gamma_events_for_cuts,
                                                        df_proton_events_for_cuts,
                                                        ntelescopes_gamma,
                                                        ntelescopes_protons,
                                                        n_bins_energy, gcut, tcut * (u.deg ** 2), noff,
                                                        fraction_of_events_for_cuts,
                                                        obstime)
                                                        
    egeom = np.sqrt(energy[1:] * energy[:-1])
    dFdE, par = crab_hegra(egeom)
    sensitivity_flux = best_sens / 100 * (dFdE * egeom * egeom).to(u.erg / (u.cm ** 2 * u.s))
    
    
    
    # Saves the results
    dl2.to_hdf(args.output_path+'/test_sens.h5', key='data')
    result.to_hdf(args.output_path+'/test_sens.h5', key='results')

    tab = Table.from_pandas(result)

    for i, key in enumerate(tab.columns.keys()):
        tab[key].unit = units[i]
        if key=='sensitivity':
            continue
        tab[key].format = '8f'
    
    
    # Plots

    fig=plt.figure(figsize=(12, 8))

    ax=plt.axes()
    plot_utils.format_axes_sensitivity(ax)
    plot_utils.plot_MAGIC_sensitivity(ax)
    plot_utils.plot_Crab_SED(ax, 100, 5, 1e5, label="100% Crab") #Energy in GeV
    plot_utils.plot_Crab_SED(ax, 10, 5, 1e5, linestyle='--', label="10% Crab") #Energy in GeV
    plot_utils.plot_Crab_SED(ax, 1, 5, 1e5, linestyle=':', label="1% Crab") #Energy in GeV
    plot_utils.plot_sensitivity(energy, best_sens, ax)
    plt.legend()
    plt.savefig(args.output_path+"/sensitivity.png")
    plt.show()
    
    plt.plot(egeom, tab['hadron_rate'], label='Hadron rate', marker='o')
    plt.plot(egeom, tab['gamma_rate'], label='Gamma rate', marker='o')
    plt.legend()
    plt.xscale('log')
    plt.xlabel('Energy (TeV)')
    plt.ylabel('events / min')
    plt.savefig(args.output_path+"/rates.png")
    plt.show()

    #fig=plt.figure(figsize=(12, 8))
    gammas_mc = dl2[dl2.mc_type == 0]
    protons_mc = dl2[dl2.mc_type == 101]
    sns.distplot(gammas_mc.gammaness, label='gammas')
    sns.distplot(protons_mc.gammaness, label='protons')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_path+"/distplot_gammaness.png")    
    plt.show()
    
    #fig=plt.figure(figsize=(12, 8))
    sns.distplot(gammas_mc.mc_energy, label='gammas');
    sns.distplot(protons_mc.mc_energy, label='protons');
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_path+"/distplot_mc_energy.png")
    plt.show()
    
    #fig=plt.figure(figsize=(12, 8))
    sns.distplot(gammas_mc.reco_energy.apply(np.log10), label='gammas')
    sns.distplot(protons_mc.reco_energy.apply(np.log10), label='protons')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_path+"/distplot_energy_apply.png")
    plt.show()
    
    #fig=plt.figure(figsize=(12, 8))
    ctaplot.plot_theta2(gammas_mc.reco_alt, gammas_mc.reco_az, gammas_mc.mc_alt, gammas_mc.mc_az, range=(0, 1), bins=100)
    plt.savefig(args.output_path+"/theta2.png")
    plt.show()
    
    #fig=plt.figure(figsize=(12, 8))
    ctaplot.plot_angular_resolution_per_energy(gammas_mc.reco_alt, gammas_mc.reco_az, gammas_mc.mc_alt, gammas_mc.mc_az, gammas_mc.reco_energy  )
    ctaplot.plot_angular_resolution_cta_requirement('north', color='black')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_path+"/angular_resolution.png")
    plt.show()
    
    #fig=plt.figure(figsize=(12, 8))
    ctaplot.plot_energy_resolution(gammas_mc.mc_energy, gammas_mc.reco_energy)
    ctaplot.plot_energy_resolution_cta_requirement('north', color='black')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_path+"/effective_area.png")
    plt.show()
    
    #fig=plt.figure(figsize=(12, 8))
    ctaplot.plot_energy_bias(gammas_mc.mc_energy, gammas_mc.reco_energy)
    plt.savefig(args.output_path+"/energy_bias.png")
    plt.show()

    #fig=plt.figure(figsize=(12, 8))
    gamma_ps_simu_info = read_simu_info_merged_hdf5(args.dl2_file_g)
    emin = gamma_ps_simu_info.energy_range_min.value
    emax = gamma_ps_simu_info.energy_range_max.value
    total_number_of_events = gamma_ps_simu_info.num_showers * gamma_ps_simu_info.shower_reuse
    spectral_index = gamma_ps_simu_info.spectral_index
    area = (gamma_ps_simu_info.max_scatter_range.value - gamma_ps_simu_info.min_scatter_range.value) ** 2 * np.pi
    ctaplot.plot_effective_area_per_energy_power_law(emin, emax, total_number_of_events, spectral_index,
                                                     gammas_mc.reco_energy[gammas_mc.tel_id == 1],
                                                     area,
                                                     label='selected gammas',
                                                     linestyle='--'
                                                     )

    ctaplot.plot_effective_area_cta_requirement('north', color='black')
    plt.ylim([2*10**3, 10**6])
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_path+"/effective_area.png")
    plt.show()
    

    '''
    #fig=plt.figure(figsize=(12, 8))
    plt.plot( energy[0:len(sensitivity_flux)], sensitivity_flux , '-', color='red', markersize=0, label='LST mono')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('$\mathsf{E^2 F \; [erg \, cm^{-2} s^{-1}]}$', fontsize = 16)
    plt.xlabel('E [TeV]')
    plt.xlim([10**-2, 100])
    plt.ylim([10**-14, 10**-9])
    plt.tight_layout()
    plt.show()
    plt.savefig('sensitivity.png')
    

    #fig=plt.figure(figsize=(12, 8))
    ctaplot.plot_energy_resolution(gammas_mc.mc_energy, gammas_mc.reco_energy, percentile=68.27, confidence_level=0.95, bias_correction=False)
    ctaplot.plot_energy_resolution_cta_requirement('north', color='black')
    plt.xscale('log')
    plt.ylabel('\u0394 E/E 68\%')
    plt.xlabel('E [TeV]')
    plt.xlim([10**-2, 100])
    plt.ylim([0.08, 0.48])
    plt.tight_layout()
    plt.savefig('energy_resolution.png', dpi=100)
    plt.show()
    '''
if __name__ == '__main__':
    main()
