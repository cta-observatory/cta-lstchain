import pandas as pd
from lstchain.tests.test_lstchain import dl2_file, dl2_params_lstcam_key
import matplotlib.pyplot as plt
import numpy as np
from lstchain.reco.utils import reco_source_position_sky
from gammapy.stats import significance_on_off

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 20

intensity_cut = 1000
leakage_cut = 0.2
wl_cut = 0.1
gammaness_cut = 0.6

dl2_file = '/fefs/aswg/workspace/maria.bernardos/LSTanalysis/real/DL2/20200118/ON/dl2_ON_merged.h5'
dl2_df = pd.read_hdf(dl2_file, key=dl2_params_lstcam_key)
dl2_file_off = '/fefs/aswg/workspace/maria.bernardos/LSTanalysis/real/DL2/20200118/OFF/dl2_OFF_merged.h5'
dl2_df_off = pd.read_hdf(dl2_file_off, key=dl2_params_lstcam_key)

obstime = 19.37

m_to_deg = 2
Tot_Non = np.shape(dl2_df)[0]
print("Total number of ON events", Tot_Non)
df_sel_gammaness = [(dl2_df['leakage'] < leakage_cut)
                    & (dl2_df['intensity'] > intensity_cut)
                    & (dl2_df['wl'] > wl_cut)
                    & (dl2_df['gammaness'] > gammaness_cut)]
df_sel_gammaness = np.array(df_sel_gammaness)
print('Number of ON events after cuts', np.sum(df_sel_gammaness[0]))

reco_src_x = dl2_df['reco_src_x'][np.array(df_sel_gammaness)[0]]
reco_src_y = dl2_df['reco_src_y'][np.array(df_sel_gammaness)[0]]
theta2 = np.power(reco_src_x * m_to_deg,2) + np.power(reco_src_y * m_to_deg,2)
theta2 = np.array(theta2)

reco_alt = dl2_df.reco_alt[np.array(df_sel_gammaness)[0]] * 180/3.14
x = dl2_df.reco_az[np.array(df_sel_gammaness)[0]]
reco_az = np.arcsin(np.sin(x)) * 180/3.14

theta2_from_alt_az = np.power(reco_alt,2) + np.power(reco_az,2)
theta2_from_alt_az = np.array(theta2_from_alt_az)

Tot_Noff = np.shape(dl2_df_off)[0]
print("Total number of OFF events", Tot_Noff)
df_sel_gammaness_off = [(dl2_df_off['leakage'] < leakage_cut)
                    & (dl2_df_off['intensity'] > intensity_cut)
                    & (dl2_df_off['wl'] > wl_cut)
                    & (dl2_df_off['gammaness'] > gammaness_cut)]
df_sel_gammaness_off = np.array(df_sel_gammaness_off)
print('Number of OFF events after cuts', np.sum(df_sel_gammaness_off[0]))

reco_src_x_off = dl2_df_off['reco_src_x'][np.array(df_sel_gammaness_off)[0]]
reco_src_y_off = dl2_df_off['reco_src_y'][np.array(df_sel_gammaness_off)[0]]
theta2_off = np.power(reco_src_x_off * m_to_deg,2) + np.power(reco_src_y_off * m_to_deg,2)
theta2_off = np.array(theta2_off)

reco_alt_off = dl2_df_off.reco_alt[np.array(df_sel_gammaness_off)[0]] * 180/3.14
x = dl2_df_off.reco_az[np.array(df_sel_gammaness_off)[0]]
reco_az_off = np.arcsin(np.sin(x)) * 180/3.14

theta2_off_from_alt_az = np.power(reco_alt_off,2) + np.power(reco_az_off,2)
theta2_off_from_alt_az = np.array(theta2_off_from_alt_az)

norm_range_th2_min = 0.5
norm_range_th2_max = 2.

Non_norm = np.sum((theta2 > norm_range_th2_min) & (theta2 < norm_range_th2_max))
Noff_norm = np.sum((theta2_off > norm_range_th2_min) & (theta2_off < norm_range_th2_max))

Norm_theta2 = Non_norm / Noff_norm
print(Norm_theta2)


theta2_cut = 0.25
#alpha = Tot_Non / Tot_Noff
Non = np.sum(theta2 < theta2_cut)
Noff = np.sum(theta2_off < theta2_cut)
Nex = Non - Noff * Norm_theta2
S = Nex / np.sqrt(Noff)
print(S)

nbins = 20
range_max = 0.5 # deg2

fig, ax = plt.subplots()
h_on = ax.hist(theta2,  label = 'ON data', bins=nbins, alpha=0.2, color = 'C3', range=[0,range_max])
h_off = ax.hist(theta2_off, weights = Norm_theta2 * np.ones(len(theta2_off)), range=[0,range_max],
                 histtype='step', label = 'OFF data', bins=nbins, alpha=0.5, color = 'k')
ax.annotate(s=f'Significance = {S:.2f} $\sigma$ \n  Rate = {Nex/obstime:.1f} $\gamma$/min',
            xy=(np.max(h_on[1]/3), np.max(h_on[0]/2)), size = 30, color = 'r')

ax.set_xlabel(r'$\theta^2$ [deg$^2$]')
ax.set_ylabel(r'Number of events')
ax.legend()

nbins = 20
range_max = 2 # deg

fig, ax = plt.subplots()
h_on = ax.hist(theta2_from_alt_az, label = 'ON data', bins=nbins, alpha=0.2, color = 'C3', range=[0,range_max])
h_off = ax.hist(theta2_off_from_alt_az, weights = Norm_theta2 * np.ones(len(theta2_off_from_alt_az)), range=[0,range_max],
                 histtype='step', label = 'OFF data', bins=nbins, alpha=0.5, color = 'k')
ax.annotate(s=f'Significance = {S:.2f} $\sigma$ \n  Rate = {Nex/obstime:.1f} $\gamma$/min',
            xy=(np.max(h_on[1]/3), np.max(h_on[0]/2)), size = 30, color = 'r')

#ax.set_yscale('log')
ax.set_xlabel(r'$\theta^2$ [deg$^2$]')
ax.set_ylabel(r'Number of events')
ax.legend()


dl2_df_off_all_cuts = [(dl2_df_off['intensity'] > intensity_cut)
                    & (dl2_df_off['leakage'] < leakage_cut)
                    & (dl2_df_off['wl'] > wl_cut)
                    & (dl2_df_off['gammaness'] > gammaness_cut)]


dl2_df_off_all_cuts = np.array(dl2_df_off_all_cuts)

cog_x_off = dl2_df_off.x[np.array(dl2_df_off_all_cuts)[0]]
cog_y_off = dl2_df_off.y[np.array(dl2_df_off_all_cuts)[0]]
psi_off = dl2_df_off.psi[np.array(dl2_df_off_all_cuts)[0]]

hip_off = np.sqrt(np.power(cog_x_off,2) + np.power(cog_y_off,2))
alpha_off = np.rad2deg(np.arccos((cog_x_off * np.cos(psi_off) + cog_y_off * np.sin(psi_off))/hip_off))
alpha_off = alpha_off * (alpha_off < 90) + (180- alpha_off) * (alpha_off >= 90)
dl2_df_all_cuts = [(dl2_df['intensity'] > intensity_cut)
                    & (dl2_df['leakage'] < leakage_cut)
                    & (dl2_df['wl'] > wl_cut)
                    & (dl2_df['gammaness'] > gammaness_cut)]


dl2_df_all_cuts = np.array(dl2_df_all_cuts)

cog_x = dl2_df.x[np.array(dl2_df_all_cuts)[0]]
cog_y = dl2_df.y[np.array(dl2_df_all_cuts)[0]]
psi = dl2_df.psi[np.array(dl2_df_all_cuts)[0]]

hip = np.sqrt(np.power(cog_x,2) + np.power(cog_y,2))
alpha = np.rad2deg(np.arccos((cog_x * np.cos(psi) + cog_y * np.sin(psi))/hip))
alpha = alpha * (alpha < 90) + (180- alpha) * (alpha >= 90)


norm_range_alpha_min = 20.
norm_range_alpha_max = 80.

Non_norm = np.sum((alpha > norm_range_alpha_min) & (alpha < norm_range_alpha_max))
Noff_norm = np.sum((alpha_off > norm_range_alpha_min) & (alpha_off < norm_range_alpha_max))

Norm_alpha = Non_norm / Noff_norm
print(Norm_alpha)

alpha_cut = 8.

#alpha = Tot_Non / Tot_Noff
Non = np.sum(alpha < alpha_cut)
Noff = np.sum(alpha_off < alpha_cut)
Nex = Non - Noff * Norm_alpha
S = Nex / (np.sqrt(Noff) * Norm_alpha)
print(S)

nbins = 30

fig, ax = plt.subplots()
h = ax.hist(alpha, label = 'ON data', bins=nbins, alpha=0.2, color = 'C3')
h2 = ax.hist(alpha_off, weights = Norm_alpha * np.ones(len(alpha_off)),
            histtype='step', label = 'OFF data', bins=nbins, alpha=0.5, color = 'k')
ax.annotate(s=f'Significance[Nex/sqrt(Nbg)] = {S:.2f} $\sigma$ \n'
            f'Rate = {Nex/obstime:.1f} $\gamma$/min \n'
            f'Nex={Nex:.1f}, Noff={Noff * Norm_alpha:.1f}',
            xy=(np.max(h[1]/3), np.max(h[0]/1.5)), size = 25, color = 'k')

ax.set_xlabel(r'$\alpha$ [deg]')
ax.set_ylabel(r'Number of events')
ax.legend()

plt.show()
