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
    total_dead_time=events.shape[0]*dead_time
    t_eff = t_elapsed/(1+rate*dead_time)

    print("ELAPSED TIME: %.2f s\n" % t_elapsed.to_value(),
          "EFFECTIVE TIME: %.2f s\n" % t_eff.to_value(),
          "DEAD TIME: %.2E s\n" % dead_time,
          "TOTAL DEAD TIME: %.2f s\n" % total_dead_time,
          "RATE: %.2f 1/s\n" % rate
    )

    return t_eff
