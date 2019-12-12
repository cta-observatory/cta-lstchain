from lstchain.mc import sensitivity
from lstchain.mc import plot_utils
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table
import numpy as np
import argparse
import ctaplot
import seaborn as sns
from lstchain.io import read_simu_info_merged_hdf5

parser = argparse.ArgumentParser(description="Compute Sensitivity Curve.")

parser.add_argument('--gammadl1', '-gd1', type=str,
                    dest='dl1file_gammas',
                    help='path to gammas simtelfile')
parser.add_argument('--protondl1', '-pd1', type=str,
                    dest='dl1file_protons',
                    help='path to protons simtelfile')
parser.add_argument('--gammadl2-cuts', '-gd2-cuts', type=str,
                    dest='dl2_file_g_cuts',
                    help='path to reconstructed gammas dl2 file')
parser.add_argument('--protondl2-cuts', '-pd2-cuts', type=str,
                    dest='dl2_file_p_cuts',
                    help='path to reconstructed protons dl2 file')
parser.add_argument('--gammadl2-sens', '-gd2-sens', type=str,
                    dest='dl2_file_g_sens',
                    help='path to reconstructed gammas dl2 file')
parser.add_argument('--protondl2-sens', '-pd2-sens', type=str,
                    dest='dl2_file_p_sens',
                    help='path to reconstructed protons dl2 file')

args = parser.parse_args()


def main():

    nfiles_gammas = 0.5  # 100*0.5 #Pointlike gammas
    nfiles_protons = 0.5  # 5000*0.8*0.5

    eb = 20  # Number of energy bins
    gb = 11  # Number of gammaness bins
    tb = 10  # Number of theta2 bins
    obstime = 50 * 3600 * u.s
    noff = 5

    E, best_sens, result, units, gcut, tcut = sensitivity.find_best_cuts_sens(args.dl1file_gammas,
                                                                              args.dl1file_protons,
                                                                              args.dl2_file_g_cuts, args.dl2_file_p_cuts,
                                                                              nfiles_gammas, nfiles_protons,
                                                                              eb, gb, tb, noff,
                                                                              obstime)
    E, best_sens, result, units, dl2 = sensitivity.sens(args.dl1file_gammas,
                                                        args.dl1file_protons,
                                                        args.dl2_file_g_sens, args.dl2_file_p_sens,
                                                        nfiles_gammas, nfiles_protons,
                                                        eb, gcut, tcut * (u.deg ** 2), noff,
                                                        obstime)
    # plt.show()
    plot_utils.sens_plot(eb, E, best_sens)
    plt.show()

    tab = Table.from_pandas(result)

    for i, key in enumerate(tab.columns.keys()):
        tab[key].unit = units[i]
        if key == 'sensitivity':
            continue
        tab[key].format = '8f'

    print(tab)
    emed = np.sqrt(E[1:] * E[:-1])

    plt.plot(emed[:-1], tab['hadron_rate'], label='Hadron rate', marker='o')
    plt.plot(emed[:-1], tab['gamma_rate'], label='Gamma rate', marker='o')
    plt.legend()
    plt.xscale('log')
    plt.xlabel('Energy (GeV)')
    plt.ylabel('events / min')
    plt.show()

    gammas_mc = dl2[dl2.mc_type == 0]
    protons_mc = dl2[dl2.mc_type == 101]
    good_gammas = dl2

    sns.distplot(gammas_mc.gammaness, label='gammas')
    sns.distplot(protons_mc.gammaness, label='protons')
    plt.legend()
    plt.tight_layout()
    plt.show()
    sns.distplot(gammas_mc.mc_energy, label='gammas')
    sns.distplot(protons_mc.mc_energy, label='protons')
    plt.legend()
    plt.tight_layout()
    plt.show()
    sns.distplot(gammas_mc.log_reco_energy, label='gammas')
    sns.distplot(protons_mc.log_reco_energy, label='protons')
    plt.legend()
    plt.tight_layout()
    plt.show()
    ctaplot.plot_theta2(gammas_mc.reco_alt, gammas_mc.reco_az, gammas_mc.mc_alt, gammas_mc.mc_az, range=(0, 1), bins=100)
    plt.show()
    plt.figure(figsize=(12, 8))
    ctaplot.plot_angular_res_per_energy(gammas_mc.reco_alt,
                                        gammas_mc.reco_az,
                                        gammas_mc.mc_alt,
                                        gammas_mc.mc_az,
                                        gammas_mc.reco_energy,
                                        )

    ctaplot.plot_angular_res_cta_requirements('north', color='black')

    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(12, 8))
    ctaplot.plot_energy_resolution(10 ** (gammas_mc.mc_energy - 3), 10 ** (gammas_mc.reco_energy - 3));

    ctaplot.plot_energy_resolution_cta_requirements('north', color='black')

    plt.legend()
    plt.tight_layout()
    plt.show()

    ctaplot.plot_energy_bias(10 ** (gammas_mc.mc_energy - 3), 10 ** (gammas_mc.reco_energy - 3))
    plt.show()

    gamma_ps_simu_info = read_simu_info_merged_hdf5(args.dl1file_gammas)
    emin = gamma_ps_simu_info.energy_range_min.value
    emax = gamma_ps_simu_info.energy_range_max.value
    total_number_of_events = gamma_ps_simu_info.num_showers * gamma_ps_simu_info.shower_reuse
    spectral_index = gamma_ps_simu_info.spectral_index
    area = (gamma_ps_simu_info.max_scatter_range.value - gamma_ps_simu_info.min_scatter_range.value) ** 2 * np.pi
    ctaplot.plot_effective_area_per_energy_power_law(emin,
                                                     emax,
                                                     total_number_of_events,
                                                     spectral_index,
                                                     gammas_mc.reco_energy[gammas_mc.tel_id == 1],
                                                     area,
                                                     label='selected gammas',
                                                     linestyle='--',
                                                     )

    ctaplot.plot_effective_area_cta_requirements('north', color='black')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
