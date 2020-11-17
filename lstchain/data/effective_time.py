import numpy as np
import astropy.units as u

def get_effecive_time(events):
    """
    Calculate the effective observation time of a set 
    of real data events.
    Parameters
    ----------
    events: pandas DataFrame

    Returns
    -------
    teff: float
    """
    
    deltat = np.diff(events.dragon_time)
    deltat = deltat[(deltat > 0) & (deltat < 0.002)]
    rate=1/np.mean(deltat)
    dead_time = np.amin(deltat)
    t_elapsed = events.shape[0]/rate * u.s
    t_eff = t_elapsed/(1+rate*dead_time)

    return t_eff
