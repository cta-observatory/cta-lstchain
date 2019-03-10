import numpy as np

def plot_Crab(ax, emin, emax, binsE, unitE, percentage = 100, SED = True, **kwargs):
    """
    Plot the Crab Nebula spectrum for sensitivity comparisons.
    
    Parameters:
    ----------
    ax: matplotlib axis object
    emin: `float` astropy units. Minimum energy
    emax: `float` astropy units. Maximum energy 
    binsE: `int` number of energy bins
    unitE: Astropy unit of energy
    percentage: `float` percentage of the Crab Nebula spectrum for the plot
    SED: `bool` if True, plot the SED, if False, plot dFdE
    kwargs for the plot

    Returns:
    ------
    ax: matplotlib axis object
    """
    E = np.logspace(np.log10(emin.to(unitE).value),np.log10(emax.to(unitE).value),binsE) * unitE
    dFdE = percentage / 100. * Crab_MAGIC(E)
    if (SED):
        ax.loglog(E, dFdE * E.to(u.TeV) * E.to(u.TeV), color='gray', **kwargs)
    else:
        ax.loglog(E, dFdE, color='gray', **kwargs)
        
    return ax
