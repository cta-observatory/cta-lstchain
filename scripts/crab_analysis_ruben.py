import pandas as pd
from lstchain.tests.test_lstchain import dl2_file, dl2_params_lstcam_key
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from lstchain.reco.utils import reco_source_position_sky
from gammapy.stats import significance_on_off

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 20

intensity_cut = 500
leakage_cut = 0.2
wl_cut = 0.1
gammaness_cut = 0.6
intensity_max_cut=80000

<<<<<<< Updated upstream
<<<<<<< Updated upstream
campaign="3rdCrabCampaign"

path="/fefs/aswg/workspace/maria.bernardos/LSTanalysis/real/DL2/"+campaign+"/v0.4.4_v00/"

if campaign=="1stCrabCampaign":

    dates=["20191123", "20191124", "20191126", "20191129"]

    runs_on=np.array([["1616", "1618", "1620", "1622"],
                      ["1627", "1629", "1631", "1632", "1633"],
                      ["1648", "1649", "1650", "1652", "1653"],
                      ["1701"]])

    runs_off=np.array([["1615", "1617", "1619", "1621"],
                       ["1626", "1628", "1630"],
                       ["1651"],
                       ["1700"]])

    runs_on_time=np.array([[24.9, 30.4, 29.1, 29.7],
                           [30., 30., 10., 10., 11.],
                           [5.0, 1.1, 23.5, 15.2, 15.7],
                           [2.5]])

    #1st campaign
    runs_off_time=np.array([[24.7, 14.6, 15.4, 15.1],
                            [16., 15., 20.],
                            [16.3, 14.2],
                            [16.3]])
if campaign=="2ndCrabCampaign":

    dates=["20200115", "20200117", "20200118", "20200127",
           "20200128", "20200201", "20200202"]

    runs_on=np.array([["1795", "1796", "1797", "1799", "1800"],
                      ["1813", "1814", "1817", "1818",
                       "1819", "1820", "1821", "1824", "1825",
                       "1826", "1827"],
                      ["1832", "1833", "1834", "1835", "1836",
                       "1843", "1844"],
                      ["1874", "1876", "1878", "1879", "1880"],
                      ["1887", "1888", "1891", "1892"],
                      ["1925", "1926", "1928", "1929",
                       "1931", "1932"],
                      ["1948", "1951", "1952", "1954", "1955"]])

    runs_off=np.array([["1798", "1801"],
                       ["1815", "1816", "1822",
                        "1823"],
                       ["1837", "1840", "1841",
                        "1842"],
                       ["1877", "1881"],
                       ["1893"],
                       ["1927", "1930", "1933"],
                       ["1949", "1953"]])

    runs_on_time=np.array([[3.10, 30.41, 19.84, 36.11, 29.65],
                           [0.21, 15.48, 17.33, 17.82,
                            23.89, 19.26, 11.87, 21.32, 18.42,
                            1.43, 0.08],
                           [19.47, 19.75, 20.87, 20.87, 20.68,
                            20.05, 36.83],
                           [20.06, 20.02, 6.7, 20.15, 19.83],
                           [0.26, 3.90, 21.04, 19.93],
                           [1.58, 19.17, 19.72, 19.93, 20.22,
                            19.28, 19.43],
                           [18.32, 22.0, 20.23, 28.42, 3.30]])

    runs_off_time=np.array([[17.49, 12.03],
                            [20.48, 18.77, 25.83,
                             13.07],
                            [20.16, 20.11, 20.08,
                             19.61],
                            [20.13, 20.30],
                            [19.63],
                            [20.15, 19.98, 18.93],
                            [10.85, 10.87]])

if campaign=="3rdCrabCampaign":

    #3rd campaign
    dates=["20200213", "20200215", "20200217", "20200218",
           "20200227", "20200228"]

    runs_on=np.array([["1970"],
                      ["1987", "1988", "1991"],
                      ["1996", "1997", "1999", "2000", "2002",
                       "2003"],
                      ["2007", "2008", "2010", "2011", "2013",
                       "2014"],
                      ["2031", "2032"],
                      ["2036", "2037"]])

    runs_off=np.array([["1971"],
                       ["1990", "1992"],
                       ["1998", "2001", "2004"],
                       ["2009", "2012"],
                       ["2033"],
                       ["2038"]])

    runs_on_time=np.array([[20.],
                           [20.08, 19.58, 20.22],
                           [20.03, 20.13, 19.70, 20.20, 22.95,
                            20.42],
                           [20.47, 20.32, 20.07, 20.92, 22.37,
                            19.95],
                           [19.02, 18.92],
                           [21.72, 19.70]])

    runs_off_time=np.array([[20.],
                            [19.15, 9.13],
                            [20.12, 19.07, 18.60],
                            [20.43, 21.50],
                            [19.33],
                            [19.75]])

=======
=======
>>>>>>> Stashed changes
path="/fefs/aswg/workspace/maria.bernardos/LSTanalysis/real/DL2/2ndCrabCampaign/"

dates=["20200115", "20200117", "20200118", "20200127",
       "20200128", "20200201", "20200202"]

runs_on=np.array([["1795", "1796", "1797", "1799", "1800"],
                  ["1813", "1814", "1817", "1818",
                   "1819", "1820", "1821", "1824", "1825",
                   "1826", "1827"],
                  ["1832", "1833", "1834", "1835", "1836",
                   "1843", "1844"],
                  ["1874", "1876", "1878", "1879", "1880"],
                  ["1887", "1888", "1891", "1892"],
                  ["1925", "1926", "1928", "1929",
                   "1931", "1932"],
                  ["1948", "1951", "1952", "1954", "1955"]])

runs_off=np.array([["1798", "1801"],
                   ["1815", "1816", "1822",
                    "1823"],
                   ["1837", "1840", "1841",
                    "1842"],
                   ["1877", "1881"],
                   ["1893"],
                   ["1927", "1930", "1933"],
                   ["1949", "1953"]])

runs_on_time=np.array([[3.10, 30.41, 19.84, 36.11, 29.65],
                       [0.21, 15.48, 17.33, 17.82,
                        23.89, 19.26, 11.87, 21.32, 18.42,
                        1.43, 0.08],
                       [19.47, 19.75, 20.87, 20.87, 20.68,
                        20.05, 36.83],
                       [20.06, 20.02, 6.7, 20.15, 19.83],
                       [0.26, 3.90, 21.04, 19.93],
                       [1.58, 19.17, 19.72, 19.93, 20.22,
                        19.28, 19.43],
                       [18.32, 22.0, 20.23, 28.42, 3.30]])

runs_off_time=np.array([[17.49, 12.03],
                        [20.48, 18.77, 25.83,
                         13.07],
                        [20.16, 20.11, 20.08,
                         19.61],
                        [20.13, 20.30],
                        [19.63],
                        [20.15, 19.98, 18.93],
                        [10.85, 10.87]])
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes

dl2_df = pd.DataFrame()
dl2_df_off = pd.DataFrame()
Tot_Non=0
Tot_Noff=0
obstime_on=0
obstime_off=0

good_dates=np.array([2])
for i, date in enumerate(dates):
#for i in good_dates:
    #date=dates[i]
    print("Adding data from %s" % date )
    for run_on in runs_on[i]:
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        dl2_file=path+date+"/"+"Run0"+run_on+"/"+"Run0"+run_on+".h5"
=======
        dl2_file=path+date+"/"+"Run"+run_on+"/"+"Run"+run_on+".h5"
>>>>>>> Stashed changes
=======
        dl2_file=path+date+"/"+"Run"+run_on+"/"+"Run"+run_on+".h5"
>>>>>>> Stashed changes
        df = pd.read_hdf(dl2_file, key=dl2_params_lstcam_key)
        df_sel_gammaness = [(df['leakage'] < leakage_cut)
                            & (df['intensity'] > intensity_cut)
                            & (df['intensity'] < intensity_max_cut)
                            & (df['wl'] > wl_cut)
                            & (df['gammaness'] > gammaness_cut)]
        Tot_Non=Tot_Non+df.shape[0]
        df = df[df_sel_gammaness[0]]
        dl2_df = dl2_df.append(df, ignore_index=True)
    obstime_on=obstime_on+sum(runs_on_time[i])

    for run_off in runs_off[i]:
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        dl2_file_off=path+date+"/"+"Run0"+run_off+"/"+"Run0"+run_off+".h5"
=======
        dl2_file_off=path+date+"/"+"Run"+run_off+"/"+"Run"+run_off+".h5"
>>>>>>> Stashed changes
=======
        dl2_file_off=path+date+"/"+"Run"+run_off+"/"+"Run"+run_off+".h5"
>>>>>>> Stashed changes
        df = pd.read_hdf(dl2_file_off, key=dl2_params_lstcam_key)
        df_sel_gammaness_off = [(df['leakage'] < leakage_cut)
                                & (df['intensity'] > intensity_cut)
                                & (df['intensity'] < intensity_max_cut)
                                & (df['wl'] > wl_cut)
                                & (df['gammaness'] > gammaness_cut)]
        Tot_Noff=Tot_Noff+df.shape[0]
        df = df[df_sel_gammaness_off[0]]
        dl2_df_off = dl2_df_off.append(df, ignore_index=True)
    obstime_off=obstime_off+sum(runs_off_time[i])

dl2_df.to_hdf("ON.h5", key=dl2_params_lstcam_key)
dl2_df_off.to_hdf("OFF.h5", key=dl2_params_lstcam_key)

#dl2_file = '/fefs/aswg/workspace/maria.bernardos/LSTanalysis/real/DL2/20200118/ON/dl2_ON_merged.h5'
#dl2_df = pd.read_hdf(dl2_file, key=dl2_params_lstcam_key)
#dl2_file_off = '/fefs/aswg/workspace/maria.bernardos/LSTanalysis/real/DL2/20200118/OFF/dl2_OFF_merged.h5'
#dl2_df_off = pd.read_hdf(dl2_file_off, key=dl2_params_lstcam_key)

obstime = obstime_on
print(obstime)
m_to_deg = 2

#Tot_Non = np.shape(dl2_df)[0]
print("Total number of ON events", Tot_Non)
'''
df_sel_gammaness = [(dl2_df['leakage'] < leakage_cut)
                    & (dl2_df['intensity'] > intensity_cut)
                    & (dl2_df['intensity'] < intensity_max_cut)
                    & (dl2_df['wl'] > wl_cut)
                    & (dl2_df['gammaness'] > gammaness_cut)]
'''
df_sel_gammaness = np.array(dl2_df)
print('Number of ON events after cuts', dl2_df.shape[0])

reco_src_x = np.array(dl2_df['reco_src_x'])
reco_src_y = np.array(dl2_df['reco_src_y'])
theta2 = np.power(reco_src_x * m_to_deg,2) + np.power(reco_src_y * m_to_deg,2)
theta2 = np.array(theta2)

reco_alt = np.array(dl2_df.reco_alt) * 180/3.14
x = np.array(dl2_df.reco_az)
reco_az = np.arcsin(np.sin(x)) * 180/3.14

theta2_from_alt_az = np.power(reco_alt,2) + np.power(reco_az,2)
theta2_from_alt_az = np.array(theta2_from_alt_az)

#Tot_Noff = np.shape(dl2_df_off)[0]
print("Total number of OFF events", Tot_Noff)
'''
df_sel_gammaness_off = [(dl2_df_off['leakage'] < leakage_cut)
                    & (dl2_df_off['intensity'] > intensity_cut)
                    & (dl2_df_off['wl'] > wl_cut)
                    & (dl2_df_off['gammaness'] > gammaness_cut)]
'''

print('Number of OFF events after cuts', dl2_df_off.shape[0])

reco_src_x_off = np.array(dl2_df_off['reco_src_x'])
reco_src_y_off = np.array(dl2_df_off['reco_src_y'])
theta2_off = np.power(reco_src_x_off * m_to_deg,2) + np.power(reco_src_y_off * m_to_deg,2)
theta2_off = np.array(theta2_off)

reco_alt_off = np.array(dl2_df_off.reco_alt) * 180/3.14
x = np.array(dl2_df_off.reco_az)
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
                 histtype='step', label = 'OFF data', bins=nbins, alpha=0.5, color = 'blue')
ax.annotate(s=f'Significance = {S:.2f} $\sigma$ \n  Rate = {Nex/obstime:.1f} $\gamma$/min \n Obs. time= {obstime/60:.1f} h',
            xy=(np.max(h_on[1]/3), np.max(h_on[0]/2)), size = 30)

ax.set_xlabel(r'$\theta^2$ [deg$^2$]')
ax.set_ylabel(r'Number of events')
ax.legend()
<<<<<<< Updated upstream
<<<<<<< Updated upstream
fig.savefig("thetaplot_%s_int%d_gammaness%.2f.pdf" % (campaign, intensity_cut, gammaness_cut))
=======
#fig.savefig("thetaplot_%s_int%d_gammaness%f.pdf" % (date, intensity_cut, gammaness_cut))
>>>>>>> Stashed changes
=======
#fig.savefig("thetaplot_%s_int%d_gammaness%f.pdf" % (date, intensity_cut, gammaness_cut))
>>>>>>> Stashed changes
nbins = 30
range_max = 2 # deg

fig, ax = plt.subplots()
h_on = ax.hist(theta2_from_alt_az, label = 'ON data', bins=nbins, alpha=0.2, color = 'C3', range=[0,range_max])
h_off = ax.hist(theta2_off_from_alt_az, weights = Norm_theta2 * np.ones(len(theta2_off_from_alt_az)), range=[0,range_max],
                 histtype='step', label = 'OFF data', bins=nbins, alpha=0.5, color = 'blue')
ax.annotate(s=f'Significance = {S:.2f} $\sigma$ \n  Rate = {Nex/obstime:.1f} $\gamma$/min \n Obs. time = {obstime/60:.1f} h',
            xy=(np.max(h_on[1]/3), np.max(h_on[0]/2)), size = 30)

#ax.set_yscale('log')
ax.set_xlabel(r'$\theta^2$ [deg$^2$]')
ax.set_ylabel(r'Number of events')
ax.legend()
<<<<<<< Updated upstream
<<<<<<< Updated upstream
fig.savefig("thetaplot_2_%s_int%d_gammaness%.2f.pdf" % (campaign, intensity_cut, gammaness_cut))
=======
#fig.savefig("thetaplot_2_int%d_gammaness%f.pdf" % (intensity_cut, gammaness_cut))
>>>>>>> Stashed changes
=======
#fig.savefig("thetaplot_2_int%d_gammaness%f.pdf" % (intensity_cut, gammaness_cut))
>>>>>>> Stashed changes

dl2_df_off_all_cuts = [(dl2_df_off['intensity'] > intensity_cut)
                       & (dl2_df['intensity'] < intensity_max_cut)
                       & (dl2_df_off['leakage'] < leakage_cut)
                       & (dl2_df_off['wl'] > wl_cut)
                       & (dl2_df_off['gammaness'] > gammaness_cut)]


dl2_df_off_all_cuts = np.array(dl2_df_off_all_cuts)

cog_x_off = np.array(dl2_df_off.x)
cog_y_off = np.array(dl2_df_off.y)
psi_off = np.array(dl2_df_off.psi)

hip_off = np.sqrt(np.power(cog_x_off,2) + np.power(cog_y_off,2))
alpha_off = np.rad2deg(np.arccos((cog_x_off * np.cos(psi_off) + cog_y_off * np.sin(psi_off))/hip_off))
alpha_off = alpha_off * (alpha_off < 90) + (180- alpha_off) * (alpha_off >= 90)

dl2_df_all_cuts = [(dl2_df['intensity'] > intensity_cut)
                   & (dl2_df['intensity'] < intensity_max_cut)
                    & (dl2_df['leakage'] < leakage_cut)
                    & (dl2_df['wl'] > wl_cut)
                    & (dl2_df['gammaness'] > gammaness_cut)]


dl2_df_all_cuts = np.array(dl2_df_all_cuts)

cog_x = np.array(dl2_df.x)
cog_y = np.array(dl2_df.y)
psi = np.array(dl2_df.psi)

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
            histtype='step', label = 'OFF data', bins=nbins, alpha=0.5, color = 'blue')
ax.annotate(s=f'Significance = {S:.2f} $\sigma$ \n'
            f'Rate = {Nex/obstime:.1f} $\gamma$/min \n'
            f'Obs. time= {obstime/60:.1f} h',
            xy=(np.max(h[1]/3), np.max(h[0]/1.5)), size = 25, color = 'k')

ax.set_xlabel(r'$\alpha$ [deg]')
ax.set_ylabel(r'Number of events')
ax.legend()
<<<<<<< Updated upstream
<<<<<<< Updated upstream
fig.savefig("alphaplot_%s_int%d_gammaness%.2f.pdf" % (campaign, intensity_cut, gammaness_cut))
=======
#fig.savefig("alphaplot_%s_int%d_gammaness%f.pdf" % (date, intensity_cut, gammaness_cut))
>>>>>>> Stashed changes
=======
#fig.savefig("alphaplot_%s_int%d_gammaness%f.pdf" % (date, intensity_cut, gammaness_cut))
>>>>>>> Stashed changes
plt.show()
