"""Module for plotting results from reconstruction.

Usage:
"import plot_dl2"
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import norm
from astropy.utils.decorators import deprecated

__all__ = [
    'plot_features',
    'plot_e',
    'plot_disp',
    'plot_disp_vector',
    'plot_pos',
    'plot_roc_gamma',
    'plot_importances',
    'plot_energy_resolution',
    'calc_resolution',
    'energy_results'
]

def plot_features(data, true_hadroness=False):
    """Plot the distribution of different features that characterize
    events, such as hillas parameters or MC data.

    Parameters:
    -----------
    data: pandas DataFrame

true_hadroness:
    True: True gammas and proton events are plotted (they are separated using true hadroness).
    False: Gammas and protons are separated using reconstructed hadroness (hadro_rec)
    """
    hadro = "reco_type"
    if true_hadroness:
        hadro = "mc_type"

    #Energy distribution
    plt.subplot(331)
    plt.hist(data[data[hadro]<1]['log_mc_energy'],
            histtype=u'step',bins=100,
             label="Gammas")
    plt.hist(data[data[hadro]>0]['log_mc_energy'],
             histtype=u'step',bins=100,
             label="Protons")
    plt.ylabel(r'# of events',fontsize=15)
    plt.xlabel(r"$log_{10}E$(GeV)")
    plt.legend()

    #disp_ distribution
    plt.subplot(332)
    plt.hist(data[data[hadro] < 1]['disp_norm'],
             histtype=u'step', bins=100,
             label="Gammas")
    plt.hist(data[data[hadro] > 0]['disp_norm'],
             histtype=u'step', bins=100,
             label="Protons")
    plt.ylabel(r'# of events', fontsize=15)
    plt.xlabel(r"disp_ (m)")

    #Intensity distribution
    plt.subplot(333)
    plt.hist(data[data[hadro] < 1]['log_intensity'],
             histtype=u'step', bins=100,
             label="Gammas")
    plt.hist(data[data[hadro] > 0]['log_intensity'],
             histtype=u'step', bins=100,
             label="Protons")
    plt.ylabel(r'# of events', fontsize=15)
    plt.xlabel(r"$log_{10}Intensity$")

    dataforwl = data[data['log_intensity'] > np.log10(200)]
    #Width distribution
    plt.subplot(334)
    plt.hist(dataforwl[dataforwl[hadro] < 1]['width'],
             histtype=u'step', bins=100,
             label="Gammas")
    plt.hist(dataforwl[dataforwl[hadro] > 0]['width'],
             histtype=u'step', bins=100,
             label="Protons")
    plt.ylabel(r'# of events', fontsize=15)
    plt.xlabel(r"Width (º)")

    #Length distribution
    plt.subplot(335)
    plt.hist(dataforwl[dataforwl[hadro] < 1]['length'],
             histtype=u'step', bins=100,
             label="Gammas")
    plt.hist(dataforwl[dataforwl[hadro] > 0]['length'],
             histtype=u'step', bins=100,
             label="Protons")
    plt.ylabel(r'# of events', fontsize=15)
    plt.xlabel(r"Length (º)")

    #r distribution
    plt.subplot(336)
    plt.hist(data[data[hadro] < 1]['r'],
             histtype=u'step', bins=100,
             label="Gammas")
    plt.hist(data[data[hadro] > 0]['r'],
             histtype=u'step', bins=100,
             label="Protons")
    plt.ylabel(r'# of events', fontsize=15)
    plt.xlabel(r"r (m)")

    #psi distribution

    plt.subplot(337)
    plt.hist(data[data[hadro] < 1]['psi'],
             histtype=u'step', bins=100,
             label="Gammas")
    plt.hist(data[data[hadro] > 0]['psi'],
             histtype=u'step', bins=100,
             label="Protons")
    plt.ylabel(r'# of events', fontsize=15)
    plt.xlabel(r"psi angle(rad)")

    #psi distribution

    plt.subplot(338)
    plt.hist(data[data[hadro] < 1]['phi'],
             histtype=u'step', bins=100,
             label="Gammas")
    plt.hist(data[data[hadro] > 0]['phi'],
             histtype=u'step', bins=100,
             label="Protons")
    plt.ylabel(r'# of events', fontsize=15)
    plt.xlabel(r"phi angle(m)")

    #Time gradient

    plt.subplot(339)
    plt.hist(data[data[hadro] < 1]['time_gradient'],
             histtype=u'step', bins=100,
             label="Gammas")
    plt.hist(data[data[hadro] > 0]['time_gradient'],
             histtype=u'step', bins=100,
             label="Protons")
    plt.ylabel(r'# of events', fontsize=15)
    plt.xlabel(r"Time gradient")


def energy_results(data):
    """
    Plot energy resolution, energy bias and energy migration matrix in the same figure

    Parameters
    ----------
    data: `pandas.DataFrame`
        dl2 MC gamma data - must include the columns `mc_energy` and `reco_energy`

    Returns
    -------
    `matplotlib.pyplot.figure`
    """
    import matplotlib
    fig, axes = plt.subplots(2, 2, figsize=(12,8))

    ctaplot.plot_energy_resolution(data.mc_energy, data.reco_energy, ax=axes[0, 0])
    ctaplot.plot_energy_bias(data.mc_energy, data.reco_energy, ax=axes[1, 0])
    ctaplot.plot_migration_matrix(data.mc_energy.apply(np.log10),
                                  data.reco_energy.apply(np.log10),
                                  ax=axes[0, 1],
                                  colorbar=True,
                                  hist2d_args=dict(norm=matplotlib.colors.LogNorm()),
                                  line_args=dict(color='black'),
                                  )
    axes[0, 1].set_xlabel('log(mc energy/[TeV])')
    axes[0, 1].set_ylabel('log(reco energy/[TeV])')
    axes[0, 0].set_title("")
    axes[0, 0].label_outer()
    axes[1, 0].set_title("")
    axes[1, 0].set_ylabel("Energy bias")
    axes[1, 1].remove()
    fig.tight_layout()
    return fig

@deprecated(message="Use energy results")
def plot_e(data, n_bins, emin, emax, true_hadroness=False):

    """Plot the performance of reconstructed Energy.

    Parameters:
    -----------
    data: pandas DataFrame

    true_hadroness:
    True: True gammas and proton events are plotted
    (they are separated using true hadroness).
    False: Gammas and protons are separated using reconstructed hadroness (hadro_rec)

    """
    hadro = "reco_type"
    if true_hadroness:
        hadro = "mc_type"

    gammas = data[data[hadro]==0]

    plt.subplot(221)

    delta_e = np.log(10**data['log_reco_energy']/10**data['log_mc_energy'])
    means_result = scipy.stats.binned_statistic(
        data['log_mc_energy'],[delta_e,delta_e**2],
        bins=n_bins,range=(emin, emax),statistic='mean')
    means, means2 = means_result.statistic
    standard_deviations = np.sqrt(means2 - means**2)
    bin_edges = means_result.bin_edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.

    plt.errorbar(x=bin_centers,
                 y=means, yerr=standard_deviations,
                 linestyle='none', marker='.')
    plt.ylabel('Bias',fontsize=24)

    plt.subplot(222)
    hE = plt.hist2d(gammas['log_mc_energy'],
                gammas['reco_energy'],
                    bins=100)

    plt.colorbar(hE[3])
    plt.xlabel('$log_{10}E_{gammas}$',
               fontsize=15)
    plt.ylabel('$log_{10}E_{rec}$',
               fontsize=15)
    plt.plot(gammas['log_mc_energy'],gammas['log_mc_energy'],
             "-",color='red')

    #Plot a profile
    plt.subplot(223)

    plt.plot(bin_centers,standard_deviations,
             marker='X',linestyle='None')
    plt.ylabel('STD',fontsize=24)
    plt.xlabel('$log_{10}E_{true}(GeV)$',fontsize=24)


    plt.subplots_adjust(hspace=.0)


def plot_disp(data, true_hadroness=False):
    """Plot the performance of reconstructed position

    Parameters:
    -----------
    data: pandas DataFrame

    true_hadroness: boolean
    True: True gammas and proton events are plotted (they are separated
    using true hadroness).
    False: Gammas and protons are separated using reconstructed
    hadroness (hadro_rec)
    """
    hadro = "reco_type"
    if true_hadroness:
        hadro = "mc_type"

    gammas = data[data[hadro] == 0]

    plt.subplot(221)

    reco_disp_norm = np.sqrt(gammas['reco_disp_dx']**2 + gammas['reco_disp_dy']**2)
    disp_res = ((gammas['disp_norm'] - reco_disp_norm) / gammas['disp_norm'])

    section = disp_res[abs(disp_res) < 0.5]
    mu,sigma = norm.fit(section)
    print("mu = {}\n sigma = {}".format(mu, sigma))

    n, bins, patches = plt.hist(disp_res,
                                bins=100,
                                density=1,
                                alpha=0.75,
                                range=[-2, 1.5],
                                )

    y = norm.pdf( bins, mu, sigma)

    plt.plot(bins, y, 'r--', linewidth=2)

    plt.xlabel('$\\frac{disp\_norm_{gammas}-disp_{rec}}{disp\_norm_{gammas}}$', fontsize=15)

    plt.figtext(0.15, 0.7, 'Mean: ' + str(round(mu, 4)), fontsize=12)
    plt.figtext(0.15, 0.65, 'Std: ' + str(round(sigma, 4)), fontsize=12)

    plt.subplot(222)

    hD = plt.hist2d(gammas['disp_norm'], reco_disp_norm,
                    bins=100,
                    range=([0, 1.1], [0, 1.1]),
                )

    plt.colorbar(hD[3])
    plt.xlabel('$disp\_norm_{gammas}$', fontsize=15)

    plt.ylabel('$disp\_norm_{rec}$', fontsize=15)

    plt.plot(gammas['disp_norm'], gammas['disp_norm'], "-", color='red')

    plt.subplot(223)
    theta2 = (gammas['src_x']-gammas['reco_src_x'])**2 + (gammas['src_y']-gammas['src_y'])**2

    plt.hist(theta2, bins=100, range=[0, 0.1], histtype=u'step')
    plt.xlabel(r'$\theta^{2}(º)$', fontsize=15)
    plt.ylabel(r'# of events', fontsize=15)


def plot_disp_vector(data):
    fig, axes = plt.subplots(1, 2)

    axes[0].hist2d(data.disp_dx, data.reco_disp_dx, bins=60);
    axes[0].set_xlabel('mc_disp')
    axes[0].set_ylabel('reco_disp')
    axes[0].set_title('disp_dx')

    axes[1].hist2d(data.disp_dy, data.reco_disp_dy, bins=60);
    axes[1].set_xlabel('mc_disp')
    axes[1].set_ylabel('reco_disp')
    axes[1].set_title('disp_dy');


def plot_pos(data,true_hadroness=False):

    """Plot the performance of reconstructed position

    Parameters:
    data: pandas DataFrame

    true_hadroness: boolean
    True: True gammas and proton events are plotted (they are separated
    using true hadroness).
    False: Gammas and protons are separated using reconstructed
    hadroness (hadro_rec)
    """
    hadro = "reco_type"
    if true_hadroness:
        hadro = "mc_type"

    #True position


    trueX = data[data[hadro]==0]['src_x']
    trueY = data[data[hadro]==0]['src_y']
    trueXprot = data[data[hadro]==101]['src_x']
    trueYprot = data[data[hadro]==101]['src_y']

    #Reconstructed position

    recX = data[data[hadro]==0]['reco_src_x']
    recY = data[data[hadro]==0]['reco_src_y']
    recXprot = data[data[hadro]==101]['reco_src_x']
    recYprot = data[data[hadro]==101]['reco_src_y']
    ran = np.array([(-0.3, 0.3), (-0.4, 0.4)])
    nbins=50

    plt.subplot(221)
    plt.hist2d(trueXprot, trueYprot,
               bins=nbins,label="Protons",
               range=ran)
    plt.colorbar()
    plt.title("True position Protons")
    plt.xlabel("x(m)")
    plt.ylabel("y (m)")

    plt.subplot(222)
    plt.hist2d(trueX,trueY,
               bins=nbins,label="Gammas",
               range=ran)
    plt.colorbar()
    plt.title("True position Gammas")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    plt.subplot(223)
    plt.hist2d(recXprot,recYprot,
               bins=nbins,label="Protons",
               range=ran)
    plt.colorbar()
    plt.title("Reconstructed position Protons")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    plt.subplot(224)
    plt.hist2d(recX,recY,
               bins=nbins,label="Gammas",
               range=ran)
    plt.colorbar()
    plt.title("Reconstructed position Gammas ")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")


def plot_importances(clf,features):

    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    print("Feature importances (gini index)")
    for f in range(len(features)):
        print("%d. %s (%f)" % (f + 1,
                               features[indices[f]],
                               importances[indices[f]]))

    ordered_features=[]
    for index in indices:
        ordered_features=ordered_features+[features[index]]

    plt.title("Feature importances",
              fontsize=15)
    plt.bar(range(len(features)),
            importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(len(features)),
               ordered_features)
    plt.xlim([-1,
              len(features)])

def calc_resolution(data):

    delta_e = np.log(10**data['reco_energy']/10**data['log_mc_energy'])
    n , bins, _ = plt.hist(delta_e,bins=500)
    mu,sigma = scipy.stats.norm.fit(delta_e)
    print(mu,sigma)
    bin_width = bins[1] - bins[0]
    total = bin_width*sum(n)*0.68
    idx = np.abs(bins - mu).argmin()
    x = 0
    mindif = 1e10
    xpos=0
    integral=0
    while integral <= total:
        integral = bin_width*sum(n[idx-x:idx+x])
        x = x+1
    print(x,integral,total)
    sigma = bins[idx+x-1]
    plt.plot(bins,integral*scipy.stats.norm.pdf(bins, mu, sigma),linewidth=4,color='red',linestyle='--')
    plt.xlabel("$log(E_{rec}/E_{true})$")
    print(mu,sigma)
    return mu,sigma

def plot_roc_gamma(dl2_data, energy_bins=None, ax=None, **kwargs):
    """
    Plot a ROC curve of the gammaness classification from a pandas dataframe.
    If there are more than two `mc_type`, all events with `mc_type!=gamma_label` are considered background.

    Parameters
    ----------
    dl2_data: `pandas.DataFrame`
        Reconstructed MC events at DL2+ level.
        must include the columns `mc_type`, `gammaness` and `mc_energy`.
    energy_bins: None or int or `numpy.ndarray`
        if None, all energy are stacked
        else, one roc curve per energy bin is done on the same plot
    ax: `matplotlib.pyplot.axis`
    kwargs: args for `ctaplot.plot_roc_curve_gammaness`

    Returns
    -------
    ax: `matplotlib.pyplot.axis`
    """
    if energy_bins is None:
        ax = ctaplot.plot_roc_curve_gammaness(dl2_data.mc_type, dl2_data.gammaness,
                                              ax=ax,
                                              **kwargs
                                              )
    else:
        ax = ctaplot.plot_roc_curve_gammaness_per_energy(dl2_data.mc_type, dl2_data.gammaness, dl2_data.mc_energy,
                                                         energy_bins=energy_bins,
                                                         ax=ax,
                                                         **kwargs)
    return ax

def plot_energy_resolution(dl2_data, ax=None, bias_correction=False, cta_req_north=False, **kwargs):
    """
    Plot the energy resolution from a pandas dataframe of DL2+ data.
    See `~ctaplot.plot_energy_resolution` for doc.

    Parameters
    ----------
   dl2_data: `pandas.DataFrame`
        Reconstructed MC events at DL2+ level.
    ax: `matplotlib.pyplot.axes`
    bias_correction: `bool`
        correct for systematic bias
    cta_req_north: `bool`
        if True, includes CTA requirement curve
    kwargs: args for `matplotlib.pyplot.plot`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = ctaplot.plot_energy_resolution(dl2_data.mc_energy,
                                dl2_data.reco_energy,
                                ax=ax,
                                bias_correction=bias_correction,
                                **kwargs,
                                )

    if cta_req_north:
        ax = ctaplot.plot_energy_resolution_cta_requirement('north', ax=ax, color='black')
    return ax

