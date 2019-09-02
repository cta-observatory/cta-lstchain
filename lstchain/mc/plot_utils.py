import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from lstchain.spectra.crab import crab_magic
from lstchain.visualization.plot_dl2 import plot_pos
import numpy as np
import astropy.units as u
import pandas as pd


__all__ = ['fill_bin_content',
           'format_axes',
           'format_axes_array',
           'format_axes_sensitivity'
           ]

def fill_bin_content(ax, sens, energy_bin, gb, tb):
    """

    Parameters
    --------

    Returns
    --------
    """
    for i in range(0,gb):
        for j in range(0,tb):
            theta2 = 0.005+0.005/2+((0.05-0.005)/tb)*j
            gammaness = 0.1/2+(1/gb)*i
            text = ax.text(theta2, gammaness, "%.2f %%" % sens[energy_bin][i][j],
                           ha="center", va="center", color="w", size=8)
    return ax

def format_axes(ax, pl):

    """
    Parameters
    --------

    Returns
    --------
    """
    ax.set_aspect('auto')

    ax.set_ylabel(r'Gammaness', fontsize=15)
    ax.set_xlabel(r'$\theta^2$ (deg$^2$)', fontsize=15)

    starty, endy = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(endy, starty, 0.1)[::-1])
    startx, endx = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(startx, endx, 0.01))

    fig = ax.get_figure()
    #cbaxes = fig.add_axes([0.9, 0.125, 0.03, 0.755])
    cbar = fig.colorbar(pl)#, cax=cbaxes)
    cbar.set_label('Sensitivity (% Crab)', fontsize=15)

def format_axes_array(ax, arr_i, arr_j, plot):
    """

    Parameters
    --------

    Returns
    --------
    """
    ax.set_aspect(0.5)
    if ((arr_i == 0) and (arr_j == 0)):
        ax.set_ylabel(r'Gammaness', fontsize=15)
    if ((arr_i == 3) and (arr_j == 2)):
        ax.set_xlabel(r'$\theta^2$ (deg$^2$)', fontsize=15)

    starty, endy = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(endy, starty, 0.1)[::-1])
    startx, endx = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(startx, endx, 0.1))

    fig = ax.get_figure()
    cbaxes = fig.add_axes([0.91, 0.125, 0.03, 0.755])
    cbar = fig.colorbar(plot, cax=cbaxes)
    cbar.set_label('Sensitivity (% Crab)', fontsize=15)


def format_axes_sensitivity(ax):
    """

    Parameters
    --------

    Returns
    --------
    """

    ax.set_xscale("log", nonposx='clip')
    ax.set_yscale("log", nonposy='clip')
    #ax.set_xlim(5e1, 9.e4)
    #ax.set_ylim(1.e-14, 5.e-10)
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel(r'E$^2$ $\frac{\mathrm{dN}}{\mathrm{dE}}$ [TeV cm$^{-2}$ s$^{-1}$]')
    ax.grid(ls='--', alpha=.5)

def plot_Crab_SED(ax, percentage, emin, emax, **kwargs):
    """

    Parameters
    --------

    Returns
    --------
    """
    En = np.logspace(np.log10(emin.to_value()), np.log10(emax.to_value()), 40) * u.GeV

    dFdE = percentage / 100. * crab_magic(En)[0]
    ax.loglog(En, (dFdE * En * En).to(u.TeV / (u.cm * u.cm * u.s)), color='gray', **kwargs)

    return ax

def plot_sensitivity(ax, e, sensitivity):
    """

    Parameters
    --------

    Returns
    --------
    """
    mask = sensitivity<1e100
    emed = np.sqrt(e[1:] * e[:-1])
    binsize = (e[1:]-e[:-1])/2

    dFdE = crab_magic(emed[mask])
    #ax.loglog(emed[mask],
    #          sensitivity[mask] / 100 * (dFdE[0] * emed[mask] * emed[mask]).to(u.TeV / (u.cm * u.cm * u.s)), label = 'Sensitivity', marker)

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.errorbar(emed[mask].to_value(), (sensitivity[mask] / 100 * (dFdE[0] * emed[mask] * emed[mask]).to(u.TeV / (u.cm * u.cm * u.s))).to_value(), xerr=binsize[mask].to_value(), marker='o')

def sens_minimization_plot(eb, gb, tb, e, sens):
    """
    TODO: Save plots!
    Parameters
    --------

    Returns
    --------
    """
    #TODO : To be changed!!!

    # if (eb == 12):
    #     figarr, axarr = plt.subplots(4,3, sharex=True, sharey=True, figsize=(13.2,18))

    figarr, axarr = plt.subplots(5,4, sharex=True, sharey=True, figsize=(13.2,18))

    # The minimum sensitivity per energy bin
    sensitivity = np.ndarray(shape=eb)

    for i in range(0, eb):
        for j in range(0, gb):
            for k in range(0, tb):
                conditions = (not np.isfinite(sens[i,j,k])) or (sens[i,j,k]<=0)
                if conditions:
                    sens[i,j,k] = 1

    for ebin in range(0,eb):
        if (figarr):
            arr_i = int(ebin/4)
            arr_j = ebin-int(ebin/4)*4
            plot = axarr[arr_i,arr_j].imshow(sens[ebin], cmap='viridis_r', \
                    extent=[0.005, 0.05, 1., 0.], norm=LogNorm(vmin=sens.min(), \
                                                vmax=sens.max()), aspect='auto')

        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_title("Ebin: %.2f - %.2f %s" % (e[ebin].to_value(),
                                               e[ebin+1].to_value(), e.unit.name))
        pl = ax.imshow(sens[ebin], cmap='viridis', extent=[0.005, 0.05, 1., 0.], aspect='auto')

        fill_bin_content(ax, sens, ebin, gb, tb)
        format_axes(ax, pl)
        fig.savefig("Ebin%d.png" % ebin)
        if (figarr):
            figarr.subplots_adjust(hspace = 0, wspace = 0)
            format_axes_array(axarr[arr_i, arr_j], arr_i, arr_j, plot)


def sens_plot(eb, e, sensitivity):
    # Final sensitivity plot
    fig_sens, ax = plt.subplots()
    plot_sensitivity(ax, e, sensitivity)

    plot_Crab_SED(ax, 100, 10**1. * u.GeV, 10**5 * u.GeV, label=r'Crab')
    plot_Crab_SED(ax, 1, 10**1. * u.GeV, 10**5 * u.GeV, ls='dotted',label='1% Crab')
    plot_Crab_SED(ax, 10, 10**1. * u.GeV, 10**5 * u.GeV, ls='-.',label='10% Crab')

    format_axes_sensitivity(ax)
    ax.legend(numpoints=1,prop={'size':9},ncol=2,loc='upper right')

def plot_positions_survived_events(df_gammas,
                                   df_protons,
                                   gammaness_g, gammaness_p,
                                   theta2_g, p_contained, sens, E, eb, g, t):

    e_reco_g = 10**df_gammas.mc_energy
    e_reco_p = 10**df_protons.mc_energy
    for i in range(0,eb):
        print(E[i], E[i+1])
        ind = np.unravel_index(np.nanargmin(sens[i], axis=None), sens[i].shape)
        events_g = df_gammas[(e_reco_g < E[i+1]) & (e_reco_g > E[i]) \
                      & (gammaness_g > g[ind[0]]) & (theta2_g < t[ind[1]])]

        events_p = df_protons[(e_reco_p < E[i+1]) & (e_reco_p > E[i]) \
                      & (gammaness_p > g[ind[0]]) & p_contained]
        events_p.intensity.hist()
        plt.xlabel("Log(10) Intensity Protons")
        plt.savefig("intensity_prot%d" % i)
        plt.show()
        df = pd.concat([events_g, events_p], ignore_index=True)
        plot_pos(df, True)
        plt.savefig("srcpos_bin%d" % i)
        plt.show()
