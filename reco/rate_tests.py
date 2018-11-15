import pyhessio
import argparse  
import numpy as np

from ctapipe.utils import get_dataset_path

parser = argparse.ArgumentParser()
parser.add_argument('--datafile', '-f', type=str,
                    dest='datafile',
                    help='path to the file with simtelarray events',
                    default=get_dataset_path('gamma_test_large.simtel.gz'))

args = parser.parse_args()

def guess_type(filename):
    """Guess the particle type from the filename
    
    Parameters
    ----------
    filename: str

    Returns
    -------
    str: 'gamma', 'proton', 'electron' or 'unknown'
    """
    particles = ['gamma', 'proton', 'electron']
    for p in particles:
        if p in filename:
            return p
    return 'unknown'

if __name__ == '__main__':
    dataset = args.datafile
    N = 0 
    Emin=-1
    Emax=-1
    index=0.
    Omega=0.
    A=0.
    Corex=0.
    Corey=0.

    particle = guess_type(dataset)

    N = pyhessio.count_mc_generated_events(dataset)
    with pyhessio.open_hessio(dataset) as f:
        f.fill_next_event()
        Emin = f.get_mc_E_range_Min()
        Emax = f.get_mc_E_range_Max()
        index = f.get_spectral_index()
        Cone = f.get_mc_viewcone_Max()
        Corex = f.get_mc_core_range_X()
        Corey = f.get_mc_core_range_Y()
    print("Number of simulated events: ",N)
    print("E min(TeV): ",Emin)
    print("E max(TeV): ",Emax)
    print("Simulated spectral index: ",index)
    print("Cone (deg)",Cone)
    print("Corex: ",Corex)
    print("Corey: ",Corey)
    
    K = N*(1+index)/(Emax**(1+index)-Emin**(1+index))
    A = np.pi*Corey**2
    Omega = 2*np.pi*(1-np.cos(Cone))
    
    MeVtoTeV = 1e-6 
    if particle=="gamma":
        K_w = 5.7e-16*MeVtoTeV
        index_w = -2.48
        E0 = 0.3e6*MeVtoTeV

    if particle=="proton":
        K_w = 9.6e-2
        index_w = -2.7
        E0 = 1

    
    Simu_E0 = K*E0**index
    N_ = Simu_E0*(Emax**(index_w+1)-Emin**(index_w+1))/(E0**index_w)/(index_w+1)
    R = K_w*A*Omega*(Emax**(index_w+1)-Emin**(index_w+1))/(E0**index_w)/(index_w+1)

    E = np.logspace(np.log10(Emin),np.log10(Emax),100)
    w = np.array([])

    for e in E:
        w_e = ((E/E0)**(index_w-index))*R/N_
        w = np.append(w,w_e)
    
    
