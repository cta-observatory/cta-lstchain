from lstchain.mc import mc
from lstchain.mc import plot_utils
from lstchain.mc import sensitivity
import numpy as np
import astropy.units as u
from lstchain.spectra.crab import crab_hegra
from lstchain.spectra.proton import proton_bess
import matplotlib.pyplot as plt

# Read files
"""
##########DIFFUSE GAMMAS###########################

simtelfile_gammas = "/fefs/aswg/workspace/MC_common/corsika6.9_simtelarray_2018-11-07/LST4_monotrigger/prod3/gamma-diffuse/gamma-diffuse_20190415/South_pointing/Data/gamma-diffuse_20deg_180deg_run98___cta-prod3-demo-2147m-LaPalma-baseline-mono.simtel.gz"

simtelfile_protons = "/fefs/aswg/workspace/MC_common/corsika6.9_simtelarray_2018-11-07/LST4_monotrigger/prod3/proton/proton_20190415/South_pointing/Data/proton_20deg_180deg_run1031___cta-prod3-demo-2147m-LaPalma-baseline-mono.simtel.gz"

dl2_file_p = "/fefs/aswg/workspace/maria.bernardos/h5files/dl2/20190415/dl2_protons_South.h5"
dl2_file_g = "/fefs/aswg/workspace/maria.bernardos/h5files/dl2/20190415/dl2_diffuse_gammas_only_South.h5"

"""
##########POINT GAMMAS############################
simtelfile_gammas = '/fefs/aswg/workspace/MC_common/corsika6.9_simtelarray_2018-11-07/LST4_monotrigger/prod3/gamma/gamma_offaxis0.4deg_20190415/South_pointing/Data/gamma_20deg_180deg_run52___cta-prod3-demo-2147m-LaPalma-baseline-mono_off0.4.simtel.gz'

simtelfile_protons = "/fefs/aswg/workspace/MC_common/corsika6.9_simtelarray_2018-11-07/LST4_monotrigger/prod3/proton/proton_20190415/South_pointing/Data/proton_20deg_180deg_run1031___cta-prod3-demo-2147m-LaPalma-baseline-mono.simtel.gz"

dl2_file_p = "/fefs/aswg/workspace/maria.bernardos/h5files/dl2/20190415/dl2_protons_forpoint_only_South.h5"
dl2_file_g = "/fefs/aswg/workspace/maria.bernardos/h5files/dl2/20190415/dl2_point_gammas_only_South.h5"

#################################################

emin_sens = 10**1. * u.GeV
emax_sens = 10**5. * u.GeV
eb = 12
gb = 10
tb = 10

obstime = obstime = 50 * 3600 * u.s

# Extract spectral parameters
E = np.logspace(np.log10(emin_sens.to_value()),
                np.log10(emax_sens.to_value()), eb + 1) * u.GeV



dFdE, crab_par = crab_hegra(E)
dFdEd0, proton_par = proton_bess(E)

# Read simulated and triggered values
gammaness_g, theta2_g, e_trig_g, mc_par_g = sensitivity.process_mc(simtelfile_gammas, dl2_file_g)
gammaness_p, theta2_p, e_trig_p, mc_par_p = sensitivity.process_mc(simtelfile_protons, dl2_file_p)

noff=mc_par_p['area_sim']/mc_par_g['area_sim']

#sim_ev: number of simulated events, it will be the number of simulated events in 1 simtel file *
# nº of files used for test (total of files * % of files used for test).

#mc_par_g['sim_ev'] = mc_par_g['sim_ev']*1000*0.2 #Diffuse gammas
mc_par_g['sim_ev'] = mc_par_g['sim_ev']*100*0.2 #Pointlike gammas
mc_par_p['sim_ev'] = mc_par_p['sim_ev']*5000*0.2

mc_par_g['emin'] = mc_par_g['emin'].to(u.GeV)
mc_par_g['emax'] = mc_par_g['emax'].to(u.GeV)

mc_par_p['emin'] = mc_par_p['emin'].to(u.GeV)
mc_par_p['emax'] = mc_par_p['emax'].to(u.GeV)

mc_par_g['area_sim'] = mc_par_g['area_sim'].to( u.cm * u.cm)
mc_par_p['area_sim'] = mc_par_p['area_sim'].to( u.cm * u.cm)

# Rates and weights
rate_g = mc.rate(mc_par_g['emin'], mc_par_g['emax'], mc_par_g['sp_idx'], \
              mc_par_g['cone'], mc_par_g['area_sim'], crab_par['f0'], crab_par['e0'])

rate_p = mc.rate(mc_par_p['emin'], mc_par_p['emax'], mc_par_p['sp_idx'], \
              mc_par_p['cone'], mc_par_p['area_sim'], proton_par['f0'], proton_par['e0'])


w_g = mc.weight(mc_par_g['emin'], mc_par_g['emax'], mc_par_g['sp_idx'],
                crab_par['alpha'], rate_g, mc_par_g['sim_ev'], crab_par['e0'])

w_p = mc.weight(mc_par_p['emin'], mc_par_p['emax'], mc_par_p['sp_idx'],
                proton_par['alpha'], rate_p, mc_par_p['sim_ev'], proton_par['e0'])


e_trig_gw = ((e_trig_g / crab_par['e0'])**(crab_par['alpha'] - mc_par_g['sp_idx'])) \
            * w_g
e_trig_pw = ((e_trig_p / proton_par['e0'])**(proton_par['alpha'] - mc_par_g['sp_idx'])) \
        * w_p

# Arrays to contain the number of gammas and hadrons for different cuts
final_gamma = np.ndarray(shape=(eb, gb, tb))
final_hadrons = np.ndarray(shape=(eb, gb, tb))
pre_gamma = np.ndarray(shape=(eb, gb, tb))
pre_hadron = np.ndarray(shape=(eb, gb, tb))

g, t = sensitivity.bin_definition(gb, tb)

for i in range(0,eb):  # binning in energy
    for j in range(0,gb):  # cut in gammaness
        for k in range(0,tb):  # cut in theta2
            eg_w_sum = np.sum(e_trig_gw[(e_trig_g < E[i+1].to_value()) & (e_trig_g > E[i].to_value()) \
                                       & (gammaness_g > g[j]) & (theta2_g < t[k])])

            ep_w_sum = np.sum(e_trig_pw[(e_trig_p < E[i+1].to_value()) & (e_trig_p > E[i].to_value()) \
                                        & (gammaness_p > g[j]) & (theta2_p < t[k])])

            pre_gamma[i][j][k] = e_trig_g[(e_trig_g < E[i+1].to_value()) & (e_trig_g > E[i].to_value()) \
                                          & (gammaness_g > g[j]) & (theta2_g < t[k])].shape[0]

            pre_hadron[i][j][k] = e_trig_p[(e_trig_p < E[i+1].to_value()) & (e_trig_p > E[i].to_value()) \
                                          & (gammaness_p > g[j]) & (theta2_p < t[k])].shape[0]

            final_gamma[i][j][k] = eg_w_sum * obstime.to_value()
            final_hadrons[i][j][k] = ep_w_sum * obstime.to_value()

sens = sensitivity.calculate_sensitivity(final_gamma, final_hadrons, 1/noff)

for i in range(0, eb):
    for j in range(0, gb):
        for k in range(0, tb):
            if np.isnan(sens[i][j][k]) or np.isinf(sens[i][j][k]) or sens[i][j][k]==0:
                sens[i][j][k] = 1e100
            if pre_gamma[i][j][k] < 10 or pre_hadron[i][j][k] < 10:
                sens[i][j][k] = 1e100

# Calculate the minimum sensitivity per energy bin
sensitivity = np.ndarray(shape=eb)
for i in range(0,eb):
    ind = np.unravel_index(np.argmin(sens[i], axis=None), sens[i].shape)
    print(E[i], g[ind[0]], t[ind[1]])
    sensitivity[i] = sens[i][ind]

#plot_utils.sens_minimization_plot(eb, gb, tb, E, sens)
mask = sensitivity<1e100

plot_utils.sens_plot(eb, E, sensitivity)
plt.show()
