import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from lstchain.spectra.crab import crab_magic
import numpy as np
import astropy.units as u

def fill_bin_content(ax, sens, energy_bin, gb, tb):
    """

    Parameters
    --------

    Returns
    --------
    """
    for i in range(0,gb):
        for j in range(0,tb):
            text = ax.text((j+0.5)*0.05, (i+0.5)*0.1, "%.2f %%" % sens[energy_bin][i][j],
                       ha="center", va="center", color="w")
    return ax

def format_axes(ax, pl):

    """

    Parameters
    --------

    Returns
    --------
    """
    ax.set_aspect(0.5)

    ax.set_ylabel(r'Gammaness', fontsize=15)
    ax.set_xlabel(r'$\theta^2$ (deg$^2$)', fontsize=15)

    starty, endy = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(endy, starty, 0.1)[::-1])
    startx, endx = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(startx, endx, 0.1))

    fig = ax.get_figure()
    cbaxes = fig.add_axes([0.9, 0.125, 0.03, 0.755])
    cbar = fig.colorbar(pl, cax=cbaxes)
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

    dFdE = crab_magic(emed[mask])
    ax.loglog(emed[mask],
              sensitivity[mask] / 100 * (dFdE[0] * emed[mask] * emed[mask]).to(u.TeV / (u.cm * u.cm * u.s)),
              label = 'Sensitivity')


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

    figarr, axarr = plt.subplots(4,3, sharex=True, sharey=True, figsize=(13.2,18))

    # The minimum sensitivity per energy bin
    sensitivity = np.ndarray(shape=eb)

    for ebin in range(0,eb):
        if (figarr):
            arr_i = int(ebin/3)
            arr_j = ebin-int(ebin/3)*3
            plot = axarr[arr_i,arr_j].imshow(sens[ebin], cmap='viridis_r', \
                   extent=[0., 0.5, 1., 0.], norm=LogNorm(vmin=sens.min(), \
                   vmax=sens.max()))

        fig, ax = plt.subplots(figsize=(8,8))

        pl = ax.imshow(sens[ebin], cmap='viridis', extent=[0., 0.5, 1., 0.])

        fill_bin_content(ax, sens, ebin, gb, tb)
        format_axes(ax, pl)

    if (figarr):
        figarr.subplots_adjust(hspace = 0, wspace = 0)
        format_axes_array(axarr[arr_i, arr_j], arr_i, arr_j, plot)


def sens_plot(eb, e, sensitivity):
    # Final sensitivity plot
    fig_sens, ax = plt.subplots()
    plot_sensitivity(ax, e, sensitivity)

    plot_Crab_SED(ax, 100, 10**1.5 * u.GeV, 10**4.5 * u.GeV, label=r'Crab')
    plot_Crab_SED(ax, 1, 10**1.5 * u.GeV, 10**4.5 * u.GeV, ls='dotted',label='1% Crab')

    format_axes_sensitivity(ax)
    ax.legend(numpoints=1,prop={'size':9},ncol=2,loc='upper right')
