from lstchain.mc import sensitivity
from lstchain.mc import plot_utils
import matplotlib.pyplot as plt
import astropy.units as u
import argparse

parser = argparse.ArgumentParser(description="Compute Sensitivity Curve.")

parser.add_argument('--gammasimtel', '-gs', type=str,
                    dest='simtelfile_gammas',
                    help='path to gammas simtelfile')
parser.add_argument('--protonsimtel', '-ps', type=str,
                    dest='simtelfile_protons',
                    help='path to protons simtelfile')
parser.add_argument('--gammadl2', '-gd2', type=str,
                    dest='dl2_file_g',
                    help='path to reconstructed gammas dl2 file')
parser.add_argument('--protondl2', '-pd2', type=str,
                    dest='dl2_file_p',
                    help='path to reconstructed protons dl2 file')

args = parser.parse_args()

nfiles_gammas = 100*0.2 #Pointlike gammas
nfiles_protons = 5000*0.2

eb = 12 # Number of energy bins
gb = 10 #Number of gammaness bins
tb = 10 #Number of theta2 bins
obstime = 50 * 3600 *u.s
noff = 5

E, best_sens = sensitivity.sens(args.simtelfile_gammas, args.simtelfile_protons,
                 args.dl2_file_g, args.dl2_file_p,
                 nfiles_gammas, nfiles_protons,
                 eb, gb, tb, noff,
                 obstime)

plot_utils.sens_plot(eb, E, best_sens)
plt.show()
