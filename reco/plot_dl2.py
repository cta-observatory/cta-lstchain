"""Module for plotting results from reconstruction.

Usage:
"import plot_dl2"
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import norm
from matplotlib import gridspec

def plot_features(data,truehadroness=False):
    """Plot the distribution of different features that characterize
    events, such as hillas parameters or MC data.

    Parameters:
    -----------
    data: pandas DataFrame
    
    truehadroness:
    True: True gammas and proton events are plotted (they are separated using true hadroness). 
    False: Gammas and protons are separated using reconstructed hadroness (Hadrorec)
    """
    hadro = "Hadrorec"
    if truehadroness:
        hadro = "hadroness"

    #Energy distribution
    plt.subplot(331)
    plt.hist(data[data[hadro]<1]['mcEnergy'],
             histtype=u'step',bins=100,
             label="Gammas")
    plt.hist(data[data[hadro]>0]['mcEnergy'],
             histtype=u'step',bins=100,
             label="Protons")
    plt.ylabel(r'# of events',fontsize=15)
    plt.xlabel(r"$log_{10}E$(GeV)")
    plt.legend()

    #Disp distribution
    plt.subplot(332)
    plt.hist(data[data[hadro]<1]['disp'],
             histtype=u'step',bins=100,
             label="Gammas")
    plt.hist(data[data[hadro]>0]['disp'],
             histtype=u'step',bins=100,
             label="Protons")
    plt.ylabel(r'# of events',fontsize=15)
    plt.xlabel(r"Disp (m)")

    #Intensity distribution
    plt.subplot(333)
    plt.hist(data[data[hadro]<1]['intensity'],
             histtype=u'step',bins=100,
             label="Gammas")
    plt.hist(data[data[hadro]>0]['intensity'],
             histtype=u'step',bins=100,
             label="Protons")
    plt.ylabel(r'# of events',fontsize=15)
    plt.xlabel(r"$log_{10}Intensity$")

    dataforwl = data[data['intensity']>np.log10(200)]
    #Width distribution
    plt.subplot(334)
    plt.hist(dataforwl[dataforwl[hadro]<1]['width'],
             histtype=u'step',bins=100,
             label="Gammas")
    plt.hist(dataforwl[dataforwl[hadro]>0]['width'],
             histtype=u'step',bins=100,
             label="Protons")
    plt.ylabel(r'# of events',fontsize=15)
    plt.xlabel(r"Width (º)")

    #Length distribution
    plt.subplot(335)
    plt.hist(dataforwl[dataforwl[hadro]<1]['length'],
             histtype=u'step',bins=100,
             label="Gammas")
    plt.hist(dataforwl[dataforwl[hadro]>0]['length'],
             histtype=u'step',bins=100,
             label="Protons")
    plt.ylabel(r'# of events',fontsize=15)
    plt.xlabel(r"Length (º)")

    #r distribution
    plt.subplot(336)
    plt.hist(data[data[hadro]<1]['r'],
             histtype=u'step',bins=100,
             label="Gammas")
    plt.hist(data[data[hadro]>0]['r'],
             histtype=u'step',bins=100,
             label="Protons")
    plt.ylabel(r'# of events',fontsize=15)
    plt.xlabel(r"r (m)")

    #psi distribution

    plt.subplot(337)
    plt.hist(data[data[hadro]<1]['psi'],
             histtype=u'step',bins=100,
             label="Gammas")
    plt.hist(data[data[hadro]>0]['psi'],
             histtype=u'step',bins=100,
             label="Protons")
    plt.ylabel(r'# of events',fontsize=15)
    plt.xlabel(r"psi angle(rad)")
    
    #psi distribution

    plt.subplot(338)
    plt.hist(data[data[hadro]<1]['phi'],
             histtype=u'step',bins=100,
             label="Gammas")
    plt.hist(data[data[hadro]>0]['phi'],
             histtype=u'step',bins=100,
             label="Protons")
    plt.ylabel(r'# of events',fontsize=15)
    plt.xlabel(r"phi angle(m)")
    
    #Time gradient

    plt.subplot(339)
    plt.hist(data[data[hadro]<1]['time_gradient'],
             histtype=u'step',bins=100,
             label="Gammas")
    plt.hist(data[data[hadro]>0]['time_gradient'],
             histtype=u'step',bins=100,
             label="Protons")
    plt.ylabel(r'# of events',fontsize=15)
    plt.xlabel(r"Time gradient")


def plot_E(data,truehadroness=False):
    
    """Plot the performance of reconstructed Energy. 

    Parameters:
    -----------
    data: pandas DataFrame
    
    truehadroness:
    True: True gammas and proton events are plotted (they are separated using true hadroness). 
    False: Gammas and protons are separated using reconstructed hadroness (Hadrorec)
    
    """
    hadro = "Hadrorec"
    if truehadroness:
        hadro = "hadroness"
    
    gammas = data[data[hadro]==0] 
    
    plt.subplot(221)
    difE = ((gammas['mcEnergy']-gammas['Erec'])*np.log(10))
    section = difE[abs(difE) < 1.5]
    mu,sigma = norm.fit(section)
    print(mu,sigma)
    n, bins, patches = plt.hist(difE,100,
                                density=1,alpha=0.75)
    y = norm.pdf( bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)
    plt.xlabel('$(log_{10}(E_{gammas})-log_{10}(E_{rec}))*log_{N}(10)$',
               fontsize=10)
    plt.figtext(0.15,0.7,'Mean: '+str(round(mu,4)),
                fontsize=10)
    plt.figtext(0.15,0.65,'Std: '+str(round(sigma,4)),
                fontsize=10)
    
    plt.subplot(222)
    hE = plt.hist2d(gammas['mcEnergy'],gammas['Erec'],
                    bins=100)
    plt.colorbar(hE[3])
    plt.xlabel('$log_{10}E_{gammas}$',
               fontsize=15)
    plt.ylabel('$log_{10}E_{rec}$',
               fontsize=15)
    plt.plot(gammas['mcEnergy'],gammas['mcEnergy'],
             "-",color='red')

    #Plot a profile
    subplot = plt.subplot(223)
    means_result = scipy.stats.binned_statistic(
        gammas['mcEnergy'],[difE,difE**2],
        bins=50,range=(1,6),statistic='mean')
    means, means2 = means_result.statistic
    standard_deviations = np.sqrt(means2 - means**2)
    bin_edges = means_result.bin_edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
    
    #fig = plt.figure()
    gs = gridspec.GridSpecFromSubplotSpec(2, 1,
                                          height_ratios=[2, 1],
                                          subplot_spec=subplot)
    ax0 = plt.subplot(gs[0])
    plot0 = ax0.errorbar(x=bin_centers, y=means, yerr=standard_deviations, 
                         linestyle='none', marker='.')
    plt.ylabel('$(log_{10}(E_{true})-log_{10}(E_{rec}))*log_{N}(10)$',
               fontsize=10)
    
    ax1 = plt.subplot(gs[1], sharex = ax0)
    plot1 = ax1.plot(bin_centers,standard_deviations,
                     marker='+',linestyle='None')
    plt.setp(ax0.get_xticklabels(), 
             visible=False)
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.ylabel("Std",fontsize=10)
    plt.xlabel('$log_{10}E_{true}$',
               fontsize=10)
    plt.subplots_adjust(hspace=.0)

    
def plot_Disp(data,truehadroness=False):
    
    """Plot the performance of reconstructed position

    Parameters:
    -----------
    data: pandas DataFrame
    
    truehadroness: boolean
    True: True gammas and proton events are plotted (they are separated
    using true hadroness). 
    False: Gammas and protons are separated using reconstructed
    hadroness (Hadrorec)
    """
    hadro = "Hadrorec"
    if truehadroness:
        hadro = "hadroness"

    gammas = data[data[hadro]==0] 

    plt.subplot(221)
    difD = ((gammas['disp']-gammas['Disprec'])/gammas['disp'])
    section = difD[abs(difD) < 0.5]
    mu,sigma = norm.fit(section)
    print(mu,sigma)
    n,bins,patches = plt.hist(difD,100,density=1,
                              alpha=0.75,range=[-2,1.5])
    y = norm.pdf( bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)
    plt.xlabel('$\\frac{Disp_{gammas}-Disp_{rec}}{Disp_{gammas}}$',
               fontsize=15)
    plt.figtext(0.15,0.7,'Mean: '+str(round(mu,4)),
                fontsize=12)
    plt.figtext(0.15,0.65,'Std: '+str(round(sigma,4)),
                fontsize=12)
                
    plt.subplot(222)
    hD = plt.hist2d(gammas['disp'],gammas['Disprec'],
                    bins=100,range=([0,1.1],[0,1.1]))
    plt.colorbar(hD[3])
    plt.xlabel('$Disp_{gammas}$',
               fontsize=15)
    plt.ylabel('$Disp_{rec}$',
               fontsize=15)
    plt.plot(gammas['disp'],gammas['disp'],
             "-",color='red')   
 
    plt.subplot(223)
    theta2 = (gammas['SrcX']-gammas['SrcXrec'])**2
    +(gammas['SrcY']-gammas['SrcY'])**2
    plt.hist(theta2,bins=100,
             range=[0,0.1],histtype=u'step')
    plt.xlabel(r'$\theta^{2}(º)$',
               fontsize=15)
    plt.ylabel(r'# of events',
               fontsize=15)
    

def plot_pos(data,truehadroness=False):
    
    """Plot the performance of reconstructed position

    Parameters:
    data: pandas DataFrame
    
    truehadroness: boolean
    True: True gammas and proton events are plotted (they are separated
    using true hadroness). 
    False: Gammas and protons are separated using reconstructed
    hadroness (Hadrorec)
    """
    hadro = "Hadrorec"
    if truehadroness:
        hadro = "hadroness"

    #True position

    trueX = data[data[hadro]==0]['SrcX']
    trueY = data[data[hadro]==0]['SrcY']
    trueXprot = data[data[hadro]==1]['SrcX']
    trueYprot = data[data[hadro]==1]['SrcY']

    #Reconstructed position

    recX = data[data[hadro]==0]['SrcXrec']
    recY = data[data[hadro]==0]['SrcYrec']
    recXprot = data[data[hadro]==1]['SrcXrec']
    recYprot = data[data[hadro]==1]['SrcYrec']

    plt.subplot(221)
    plt.hist2d(trueXprot,trueYprot,
               bins=100,label="Protons")
    plt.colorbar()
    plt.title("True position Protons")
    plt.xlabel("x(m)")
    plt.ylabel("y (m)")
    plt.subplot(222)
    plt.hist2d(trueX,trueY,
               bins=100,label="Gammas",
               range=np.array([(-1, 1), (-1, 1)]))
    plt.colorbar()
    plt.title("True position Gammas")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.subplot(223)
    plt.hist2d(recXprot,recYprot,
               bins=100,label="Protons",
               range=np.array([(-1, 1), (-1, 1)]))
    plt.colorbar()
    plt.title("Reconstructed position Protons")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.subplot(224)
    plt.hist2d(recX,recY,
               bins=100,label="Gammas",
               range=np.array([(-1, 1), (-1, 1)]))
    plt.colorbar()
    plt.title("Reconstructed position Gammas ")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
