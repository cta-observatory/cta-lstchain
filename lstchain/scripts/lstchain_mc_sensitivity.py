#!/usr/bin/env python3

"""
Script to compute the LST sensitivity using MC.

Inputs are DL1/DL2 gamma and proton files

Usage: 

$> python lstchain_mc_sensitivity.py
--gd1 dl1_gamma_20deg_180deg_cta-prod3-demo-2147m-LaPalma-baseline-mono_off0.4_merge_test.h5 
--pd1 dl1_proton_20deg_180degcta-prod3-demo-2147m-LaPalma-baseline-mono_merge_test.h5 
--gd2-cuts dl2_gammas_cuts.h5  
--pd2-cuts dl2_protons_cuts.h5 
--gd2-sens dl2_gammas_sensitivity.h5 
--pd2-sens dl2_protons_sensitivity.h5

"""


from lstchain.mc.sensitivity import sensitivity, find_best_cuts_sensitivity
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table
import numpy as np
import argparse
import ctaplot
from lstchain.visualization import plot_dl2
from lstchain.reco import utils
import seaborn as sns
from lstchain.io import read_simu_info_merged_hdf5
from lstchain.spectra.crab import crab_hegra

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=RuntimeWarning)

ctaplot.set_style()


parser = argparse.ArgumentParser(description="Compute MC sensitivity curve.")

parser.add_argument('--input-file-gamma-dl1', '--gd1', type = str,
                    dest = 'dl1file_gammas',
                    help = 'path to gammas DL1 file')
parser.add_argument('--input-file-proton-dl1', '--pd1', type = str,
                    dest = 'dl1file_protons',
                    help = 'path to protons DL1 file')
parser.add_argument('--input-file-gamma-dl2-cuts', '--gd2-cuts', type = str,
                    dest = 'dl2_file_g_cuts',
                    help = 'path to reconstructed gammas dl2 file used to calculate the sensitivity')
parser.add_argument('--input-file-proton-dl2-cuts', '--pd2-cuts', type = str,
                    dest = 'dl2_file_p_cuts',
                    help = 'path to reconstructed protons dl2 file used to calculate the sensitivity')
parser.add_argument('--input-file-gamma-dl2-sens', '--gd2-sens', type = str,
                    dest = 'dl2_file_g_sens',
                    help = 'path to reconstructed gammas dl2 file used to calculate the optimized cuts' 
                    ' to be applied to the sensitivity')
parser.add_argument('--input-file-proton-dl2-sens', '--pd2-sens', type = str,
                    dest = 'dl2_file_p_sens',
                    help = 'path to reconstructed protons dl2 file used to calculate the optimized cuts'
                    ' to be applied to the sensitivity')

args = parser.parse_args()


def main():
    ntelescopes_gamma = 4
    ntelescopes_protons = 1
    n_bins_energy = 20  #  Number of energy bins
    n_bins_gammaness = 10  #  Number of gammaness bins
    n_bins_theta2 = 10  #  Number of theta2 bins
    obstime = 50 * 3600 * u.s
    noff = 5
    
    # Finds the best cuts for the computation of the sensitivity
    '''energy, best_sens, result, units, gcut, tcut = find_best_cuts_sensitivity(args.dl1file_gammas,
                                                                              args.dl1file_protons,
                                                                              args.dl2_file_g_sens,
                                                                              args.dl2_file_p_sens,
                                                                              ntelescopes_gamma, ntelescopes_protons,
                                                                              n_bins_energy, n_bins_gammaness,
                                                                              n_bins_theta2, noff,
                                                                              obstime)
    '''                                                                          
    #For testing using fixed cuts
    gcut = np.ones(n_bins_energy) * 0.8 
    tcut = np.ones(n_bins_energy) * 0.01
    
    print("\nApplying optimal gammaness cuts:", gcut)
    print("Applying optimal theta2 cuts: {} \n".format(tcut))


    # Computes the sensitivity
    energy, best_sens, result, units, dl2 = sensitivity(args.dl1file_gammas,
                                                        args.dl1file_protons,
                                                        args.dl2_file_g_cuts, 
                                                        args.dl2_file_p_cuts,
                                                        1, 1,
                                                        n_bins_energy, gcut, tcut * (u.deg ** 2), noff,
                                                        obstime)
                                                        
    egeom = np.sqrt(energy[1:] * energy[:-1])
    dFdE, par = crab_hegra(egeom)
    sensitivity_flux = best_sens / 100 * (dFdE * egeom * egeom).to(u.erg / (u.cm ** 2 * u.s))
    
    
    
    # Saves the results
    dl2.to_hdf('test_sens.h5', key='data')
    result.to_hdf('test_sens.h5', key='results')

    tab = Table.from_pandas(result)

    for i, key in enumerate(tab.columns.keys()):
        tab[key].unit = units[i]
        if key=='sensitivity':
            continue
        tab[key].format = '8f'
    
    
    # Plots
    
    plt.figure(figsize=(12, 8))
    plt.plot(egeom[:-1], tab['hadron_rate'], label='Hadron rate', marker='o')
    plt.plot(egeom[:-1], tab['gamma_rate'], label='Gamma rate', marker='o')
    plt.legend()
    plt.xscale('log')
    plt.xlabel('Energy (TeV)')
    plt.ylabel('events / min')
    plt.show()
    plt.savefig("rates.png")
    
    plt.figure(figsize=(12, 8))
    gammas_mc = dl2[dl2.mc_type == 0]
    protons_mc = dl2[dl2.mc_type == 101]
    sns.distplot(gammas_mc.gammaness, label='gammas')
    sns.distplot(protons_mc.gammaness, label='protons')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("distplot_gammaness.png")    
    
    plt.figure(figsize=(12, 8))
    sns.distplot(gammas_mc.mc_energy, label='gammas');
    sns.distplot(protons_mc.mc_energy, label='protons');
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("distplot_mc_energy.png")
    
    plt.figure(figsize=(12, 8))
    sns.distplot(gammas_mc.reco_energy.apply(np.log10), label='gammas')
    sns.distplot(protons_mc.reco_energy.apply(np.log10), label='protons')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("distplot_energy_apply.png")
    
    plt.figure(figsize=(12, 8))
    ctaplot.plot_theta2(gammas_mc.reco_alt, gammas_mc.reco_az, gammas_mc.mc_alt, gammas_mc.mc_az, range=(0, 1), bins=100)
    plt.show()
    plt.savefig("theta2.png")
    
    plt.figure(figsize=(12, 8))
    ctaplot.plot_angular_resolution_per_energy(gammas_mc.reco_alt, gammas_mc.reco_az, gammas_mc.mc_alt, gammas_mc.mc_az, gammas_mc.reco_energy  )
    ctaplot.plot_angular_resolution_cta_requirement('north', color='black')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("angular_resolution.png")
    
    plt.figure(figsize=(12, 8))
    ctaplot.plot_energy_resolution(gammas_mc.mc_energy, gammas_mc.reco_energy)
    ctaplot.plot_energy_resolution_cta_requirement('north', color='black')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("effective_area.png")

    plt.figure(figsize=(12, 8))
    ctaplot.plot_energy_bias(gammas_mc.mc_energy, gammas_mc.reco_energy)
    plt.show()
    plt.savefig("energy_bias.png")
    
    plt.figure(figsize=(12, 8))
    gamma_ps_simu_info = read_simu_info_merged_hdf5(args.dl1file_gammas)
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
    plt.show()
    plt.savefig("effective_area.png")
    
    plt.figure(figsize=(12, 8))
    plt.plot( energy[0:len(sensitivity_flux)], sensitivity_flux , '-', color='red', markersize=0, label='LST mono')
    plt.xscale('log')
    plt.yscale('log')

    plt.ylabel('$\mathsf{E^2 F \; [erg \, cm^{-2} s^{-1}]}$', fontsize = 16)
    plt.xlabel('E [TeV]')
    plt.xlim([10**-2, 100])
    plt.ylim([10**-14, 10**-9])
    plt.tight_layout()
    plt.savefig('sensitivity.png')


    plt.figure(figsize=(12, 8))
    ctaplot.plot_energy_resolution(gammas_mc.mc_energy, gammas_mc.reco_energy, percentile=68.27, confidence_level=0.95, bias_correction=False)
    ctaplot.plot_energy_resolution_cta_requirement('north', color='black')
    plt.xscale('log')
    plt.ylabel('\u0394 E/E 68\%')
    plt.xlabel('E [TeV]')
    plt.xlim([10**-2, 100])
    plt.ylim([0.08, 0.48])
    plt.tight_layout()

    plt.savefig('energy_resolution.png', dpi=100)

if __name__ == '__main__':
    main()
