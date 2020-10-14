#!/usr/bin/env python3

"""
Script to compute the LST sensitivity using MC.

Inputs are DL1/DL2 gamma and proton files

Usage:

$> python lstchain_mc_sensitivity.py
--gdl2 dl2_gammas.h5
--pdl2 dl2_protons.h5
--o /output/path

"""


from lstchain.mc.sensitivity import sensitivity_gamma_efficiency, sensitivity_gamma_efficiency_real_protons
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table
import numpy as np
import argparse
import ctaplot
from lstchain.reco import utils
import seaborn as sns
from lstchain.io import read_simu_info_merged_hdf5
from lstchain.spectra.crab import crab_hegra
from lstchain.mc import plot_utils
import os

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
    obstime = 50 * 3600 * u.s
    noff = 5
    geff_gammaness = 1. #Gamma efficincy of gammaness cut
    geff_theta2 = 1. #Gamma efficiency of theta2 cut


    # Calculate the sensitivity
    '''
    energy,sensitivity,result,events, gcut, tcut = sensitivity_gamma_efficiency(args.dl2_file_g,
                                                                                         args.dl2_file_p,
                                                                                         ntelescopes_gamma,
                                                                                         ntelescopes_protons,
                                                                                         n_bins_energy,
                                                                                         geff_gammaness,
                                                                                         geff_theta2,
                                                                                         noff,
                                                                                         obstime)

    '''

    energy,sensitivity,result,events, gcut, tcut = sensitivity_gamma_efficiency_real_protons(args.dl2_file_g,
                                                                                             args.dl2_file_p,
                                                                                             ntelescopes_gamma,
                                                                                             n_bins_energy,
                                                                                             geff_gammaness,
                                                                                             geff_theta2,
                                                                                             noff,
                                                                                             obstime)

    print("\nOptimal gammaness cuts:", gcut)
    print("Optimal theta2 cuts: {} \n".format(tcut))

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Saves the results
    events.to_hdf(args.output_path+'/sensitivity.h5', key='data', mode='w')
    result.to_hdf(args.output_path+'/sensitivity.h5', key='results')

    # Plots

    #Sensitivity
    ax=plt.axes()
    plot_utils.format_axes_sensitivity(ax)
    plot_utils.plot_MAGIC_sensitivity(ax, color='C0')
    plot_utils.plot_LST_preliminary_sensitivity(ax, color='C2')
    plot_utils.plot_Crab_SED(ax, 100, 50, 5e4, label="100% Crab") #Energy in GeV
    plot_utils.plot_Crab_SED(ax, 10, 50, 5e4, linestyle='--', label="10% Crab") #Energy in GeV
    plot_utils.plot_Crab_SED(ax, 1, 50, 5e4, linestyle=':', label="1% Crab") #Energy in GeV
    plot_utils.plot_sensitivity(energy, sensitivity, ax)
    plt.legend(prop={'size': 12})
    plt.savefig(args.output_path+"/sensitivity.png")
    plt.show()

    #Rates

    egeom = np.sqrt(energy[1:] * energy[:-1])
    plt.plot(egeom, result['proton_rate'], label='Proton rate', marker='o')
    plt.plot(egeom, result['gamma_rate'], label='Gamma rate', marker='o')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Energy (TeV)')
    plt.ylabel('events / min')
    plt.savefig(args.output_path+"/rates.png")
    plt.show()

    #Gammaness
    gammas_mc = events[events.mc_type == 0]
    protons_mc = events[events.mc_type != 0]
    sns.distplot(gammas_mc.gammaness, label='gammas')
    sns.distplot(protons_mc.gammaness, label='protons')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_path+"/distplot_gammaness.png")
    plt.show()

    #True Energy
    sns.distplot(gammas_mc.mc_energy, label='gammas');
    sns.distplot(protons_mc.mc_energy, label='protons');
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_path+"/distplot_mc_energy.png")
    plt.show()


    #Reconstructed Energy
    sns.distplot(gammas_mc.reco_energy.apply(np.log10), label='gammas')
    sns.distplot(protons_mc.reco_energy.apply(np.log10), label='protons')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_path+"/distplot_energy_apply.png")
    plt.show()


    #Theta2
    ctaplot.plot_theta2(gammas_mc.reco_alt, gammas_mc.reco_az, gammas_mc.mc_alt, gammas_mc.mc_az, range=(0, 1), bins=100)
    plt.savefig(args.output_path+"/theta2.png")
    plt.show()

    #Angular resolution
    ctaplot.plot_angular_resolution_per_energy(events.reco_alt, events.reco_az, events.mc_alt, events.mc_az, events.reco_energy  )
    ctaplot.plot_angular_resolution_cta_requirement('north', color='black')

    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_path+"/angular_resolution.png")
    plt.show()

    #Energy resolution

    ctaplot.plot_energy_resolution(events.mc_energy, events.reco_energy)
    ctaplot.plot_energy_resolution_cta_requirement('north', color='black')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_path+"/effective_area.png")
    plt.show()

    #Energy bias

    ctaplot.plot_energy_bias(events.mc_energy, events.reco_energy)
    plt.savefig(args.output_path+"/energy_bias.png")
    plt.show()

    #Effective Area

    gamma_ps_simu_info = read_simu_info_merged_hdf5(args.dl2_file_g)
    emin = gamma_ps_simu_info.energy_range_min.value
    emax = gamma_ps_simu_info.energy_range_max.value
    total_number_of_events = gamma_ps_simu_info.num_showers * gamma_ps_simu_info.shower_reuse * ntelescopes_gamma
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

if __name__ == '__main__':
    main()
