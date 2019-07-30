import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from lstchain.spectra.crab import crab_magic, crab_hegra
from lstchain.visualization.plot_dl2 import plot_pos
import numpy as np
import astropy.units as u
import pandas as pd

__all__ = ['fill_bin_content',
           'format_axes_ebin',
           'format_axes_array',
           'format_axes_sensitivity',
           'plot_MAGIC_sensitivity',
           'plot_Crab_SED',
           'plot_sensitivity',
           'sens_minimization_plot',
           'sensitivity_plot_comparison',
           'plot_positions_survived_events',
           ]

def fill_bin_content(ax, sens, energy_bin, gb, tb):
    """
    Function to fill bin content to be plotted in the case of an
    optimized figure array

    Parameters
    --------
    ax:    `matplotlib.pyplot.axis`  
    sens:  `numpy.ndarray`  sensitivity array
    energy_bin:    `int`  energy bin number
    gb:    `int`  number of bins in gammaness
    tb:    `int`  number of bins in theta2

    Returns
    --------
    ax:    `matplotlib.pyplot.axis`  

    """

    for i in range(0, gb):
        for j in range(0, tb):
            theta2 = 0.005 + 0.005 / 2 + ((0.05 - 0.005)/tb) * j
            gammaness = 0.1/2+(1/gb) * i
            text = ax.text(theta2, gammaness, "%.2f %%" % sens[energy_bin][i][j],
                           ha = "center", va = "center", color = "w", size = 8)
    return ax

def format_axes_ebin(ax, pl):
    """
    Format axes for the theta2 and gammaness optimization per energy bin

    Parameters
    --------
    ax:    `matplotlib.pyplot.axis`  
    pl:    `matplotlib.pyplot.figure`  

    Returns
    --------
    ax:    `matplotlib.pyplot.axis`  

    """

    ax.set_aspect('auto')

    ax.set_ylabel(r'Gammaness', fontsize = 15)
    ax.set_xlabel(r'$\theta^2$ (deg$^2$)', fontsize = 15)

    starty, endy = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(endy, starty, 0.1)[::-1])
    startx, endx = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(startx, endx, 0.01))

    fig = ax.get_figure()
    #cbaxes = fig.add_axes([0.9, 0.125, 0.03, 0.755])
    cbar = fig.colorbar(pl)#, cax=cbaxes)
    cbar.set_label('Sensitivity (% Crab)', fontsize = 15)

def format_axes_array(ax, arr_i, arr_j, plot):
    """
    Format axes for the theta2 and gammaness optimization for a
    figure array with all energy bins together

    Parameters
    --------
    ax:    `matplotlib.pyplot.axis`  
    arr_i: `int` i index for the square plot  
    arr_j: `int` j index for the square plot
    plot:  `matplotlib.pyplot.figure`  

    Returns
    --------
    ax:    `matplotlib.pyplot.axis`  
    """
    ax.set_aspect(0.5)
    if ((arr_i == 0) and (arr_j == 0)):
        ax.set_ylabel(r'Gammaness', fontsize=15)
    if ((arr_i == 3) and (arr_j == 2)):
        ax.set_xlabel(r'$\theta^2$ (deg$^2$)', fontsize = 15)

    starty, endy = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(endy, starty, 0.1)[::-1])
    startx, endx = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(startx, endx, 0.1))

    fig = ax.get_figure()
    cbaxes = fig.add_axes([0.91, 0.125, 0.03, 0.755])
    cbar = fig.colorbar(plot, cax = cbaxes)
    cbar.set_label('Sensitivity (% Crab)', fontsize = 15)


def format_axes_sensitivity(ax):
    """
    Format axes of the sensitivity plot

    Parameters
    --------
    ax:    `matplotlib.pyplot.axis`  

    Returns
    --------
    ax:    `matplotlib.pyplot.axis`  

    """

    ax.set_xscale("log", nonposx = 'clip')
    ax.set_yscale("log", nonposy = 'clip')
    #ax.set_xlim(5e1, 9.e4)
    #ax.set_ylim(1.e-14, 5.e-10)
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel(r'E$^2$ $\frac{\mathrm{dN}}{\mathrm{dE}}$ [TeV cm$^{-2}$ s$^{-1}$]')
    ax.grid(ls='--', alpha = .5)

def plot_MAGIC_sensitivity(ax):
    """
    Plot MAGIC sensitivity for comparison with the reached one

    Parameters
    --------
    ax:    `matplotlib.pyplot.axis`  

    Returns
    --------
    ax:    `matplotlib.pyplot.axis`  

    """

    s = np.loadtxt('../spectra/data/magic_sensitivity.txt', skiprows = 1)   
    ax.loglog(s[:,0], s[:,3] * np.power(s[:,0] / 1.e3, 2), 
              color = 'C0', label = 'MAGIC (Aleksic et al. 2014)')
    
    return ax

def plot_Crab_SED(ax, percentage, emin, emax, **kwargs):
    """
    Plot a percentage of the Crab SED to compare with the achieved
    sensitivity

    Parameters
    --------
    ax:    `matplotlib.pyplot.axis`  
    percentage:    `float`  percentage of the Crab Nebula to be plotted
    emin:   `float` minimum energy
    emax:   `float` maximum energy

    Returns
    --------
    ax:    `matplotlib.pyplot.axis`  

    """

    En = np.logspace(np.log10(emin.to_value()), np.log10(emax.to_value()), 40) * u.GeV

    dFdE = percentage / 100. * crab_magic(En)[0]
    ax.loglog(En, (dFdE * En * En).to(u.TeV / (u.cm * u.cm * u.s)), color = 'gray', **kwargs)

    return ax

def plot_sensitivity(ax, e, sensitivity):
    """
    Plot the achieved sensitivity

    Parameters
    --------
    ax:    `matplotlib.pyplot.axis`  
    e:           `numpy.ndarray`  energy array
    sensitivity: `numpy.ndarray`  sensitivity array (bins of energy)

    Returns
    --------
    ax:    `matplotlib.pyplot.axis`  

    """

    mask = sensitivity < 1e100
    egeom = np.sqrt(e[1:] * e[:-1])
    binsize = (e[1:] - e[:-1]) / 2

    dFdE = crab_hegra(egeom[mask])

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.errorbar(egeom[mask].to_value(), 
                (sensitivity[mask] / 100 * (dFdE[0] * egeom[mask] *\ 
                                            egeom[mask]).to(u.TeV / (u.cm * u.cm * u.s))).to_value(), 
                xerr=binsize[mask].to_value(), marker = 'o', color = 'C3', label = 'Sensitivity')

def sens_minimization_plot(eb, gb, tb, e, sens):
    """
    Plot the sensitivity minimization plots in different
    energy bins to check that the theta2 and gammaness
    cuts were properly applied

    TODO: Save plots!
    Parameters
    --------
    eb:    `int`  number of bins in energy
    gb:    `int`  number of bins in gammaness
    tb:    `int`  number of bins in theta2
    e:  `numpy.ndarray`  sensitivity array
    sens: `numpy.ndarray`  sensitivity array (bins of energy, gammaness and theta2)

    Returns
    --------
    figarr: `matplotlib.pyplot.figure`   

    """

    #TODO : To be changed!!!
    # if (eb == 12):
    #     figarr, axarr = plt.subplots(4,3, sharex=True, sharey=True, figsize=(13.2,18))

    figarr, axarr = plt.subplots(5,4, sharex = True, sharey = True, figsize = (13.2,18))

    # The minimum sensitivity per energy bin
    sensitivity = np.ndarray(shape = eb)

    for i in range(0, eb):
        for j in range(0, gb):
            for k in range(0, tb):
                conditions = (not np.isfinite(sens[i,j,k])) or (sens[i,j,k] <= 0)
                if conditions:
                    sens[i,j,k] = 1

    for ebin in range(0,eb):
        if (figarr):
            arr_i = int(ebin / 4)
            arr_j = ebin-int(ebin / 4) * 4
            plot = axarr[arr_i,arr_j].imshow(sens[ebin], cmap = 'viridis_r', \
                    extent=[0.005, 0.05, 1., 0.], norm = LogNorm(vmin=sens.min(), \
                                                vmax = sens.max()), aspect = 'auto')

        fig, ax = plt.subplots(figsize = (8, 8))
        ax.set_title("Ebin: %.2f - %.2f %s" % (e[ebin].to_value(),
                                               e[ebin+1].to_value(), e.unit.name))
        pl = ax.imshow(sens[ebin], cmap='viridis', extent = [0.005, 0.05, 1., 0.], aspect = 'auto')

        fill_bin_content(ax, sens, ebin, gb, tb)
        format_axes_ebin(ax, pl)
        fig.savefig("Ebin%d.png" % ebin)
        if (figarr):
            figarr.subplots_adjust(hspace = 0, wspace = 0)
            format_axes_array(axarr[arr_i, arr_j], arr_i, arr_j, plot)

    return figarr


def sensitivity_plot_comparison(eb, e, sensitivity):
    """
    Main sensitivity plot.
    We plot the sensitivity achieved, MAGIC sensitivity and Crab SEDs

    Parameters
    --------
    eb:   `int`  number of bins in energy
    e:    `numpy.ndarray`  sensitivity array
    sens: `numpy.ndarray`  sensitivity array (bins of energy, gammaness and theta2)

    Returns
    --------
    fig_sens: `matplotlib.pyplot.figure` Figure containing sensitivity plot

    """

    # Final sensitivity plot
    fig_sens, ax = plt.subplots()
    plot_sensitivity(ax, e, sensitivity)

    plot_Crab_SED(ax, 100, 10**1. * u.GeV, 10**5 * u.GeV, label = r'Crab')
    plot_Crab_SED(ax, 1, 10**1. * u.GeV, 10**5 * u.GeV, ls = 'dotted',label = '1% Crab')
    plot_Crab_SED(ax, 10, 10**1. * u.GeV, 10**5 * u.GeV, ls = '-.',label = '10% Crab')

    plot_MAGIC_sensitivity(ax)
    format_axes_sensitivity(ax)
    ax.legend(numpoints = 1, prop = {'size':9}, ncol = 2, loc = 'upper right')

    return fig_sens

def plot_positions_survived_events(df_gammas,
                                   df_protons,
                                   gammaness_g, gammaness_p,
                                   theta2_g, p_contained, sens, E, eb, g, t):
    """
    Plot positions of surviving events after cuts

    Parameters
    --------
    df_gammas: `pandas.DataFrame` gammas dl2 parameters
    df_protons: `pandas.DataFrame` protons dl2 parameters
    gammaness_g: `numpy.ndarray`  gammaness array of gamma events
    gammaness_p: `numpy.ndarray`  gammaness array of proton events
    theta2_g: `numpy.ndarray`  theta2 array of gamma events
    p_contained: `numpy.ndarray`  containment of proton events inside
                  the ring established in camera coordinates
    sens: `numpy.ndarray`  array with sensitivity values in energy bins
    E: `numpy.ndarray`  energy edge bins (size eb+1)
    eb: `int`  number of bins in energy
    gammaness: `numpy.ndarray`  gammaness bins
    theta2: `numpy.ndarray`  theta2 bins

    Returns
    --------

    """

    e_reco_g = 10**df_gammas.mc_energy
    e_reco_p = 10**df_protons.mc_energy
    for i in range(0,eb):
        fig, ax = plt.subplots()
        print("Energy range [GeV]: ", E[i], E[i+1])
        ind = np.unravel_index(np.nanargmin(sens[i], axis = None), sens[i].shape)
        events_g = df_gammas[(e_reco_g < E[i+1]) & (e_reco_g > E[i]) \
                      & (gammaness_g > gammaness[ind[0]]) & (theta2_g < theta2[ind[1]])]

        events_p = df_protons[(e_reco_p < E[i+1]) & (e_reco_p > E[i]) \
                      & (gammaness_p > gammaness[ind[0]]) & p_contained]
        events_p.intensity.hist()
        ax.xlabel("Log(10) Intensity Protons")
        ax.savefig("intensity_prot%d" % i)
        # ax.show()
        df = pd.concat([events_g, events_p], ignore_index = True)
        plot_pos(df, True)
        fig.savefig("srcpos_bin%d" % i)
        # ax.show()
