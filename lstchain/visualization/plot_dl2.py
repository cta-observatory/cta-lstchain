"""Module for plotting results from reconstruction.

Usage:
"import plot_dl2"
"""
import os

import astropy.units as u
import ctaplot
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io.misc.hdf5 import write_table_hdf5, read_table_hdf5
from astropy.table import Table
from matplotlib.cm import get_cmap
from scipy.stats import norm

from ..io.config import get_standard_config, read_configuration_file

__all__ = [
    'direction_results',
    'energy_results',
    'plot_disp',
    'plot_disp_vector',
    'plot_energy_resolution',
    'plot_features',
    'plot_importances',
    'plot_pos',
    'plot_roc_gamma',
    'plot_1d_excess',
    'plot_wobble',
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

    # Energy distribution
    plt.subplot(331)
    plt.hist(data[data[hadro] < 1]['log_mc_energy'],
             histtype=u'step', bins=100,
             label="Gammas")
    plt.hist(data[data[hadro] > 0]['log_mc_energy'],
             histtype=u'step', bins=100,
             label="Protons")
    plt.ylabel(r'# of events', fontsize=15)
    plt.xlabel(r"$log_{10}E$(GeV)")
    plt.legend()

    # disp_ distribution
    plt.subplot(332)
    plt.hist(data[data[hadro] < 1]['disp_norm'],
             histtype=u'step', bins=100,
             label="Gammas")
    plt.hist(data[data[hadro] > 0]['disp_norm'],
             histtype=u'step', bins=100,
             label="Protons")
    plt.ylabel(r'# of events', fontsize=15)
    plt.xlabel(r"disp_ (m)")

    # Intensity distribution
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
    # Width distribution
    plt.subplot(334)
    plt.hist(dataforwl[dataforwl[hadro] < 1]['width'],
             histtype=u'step', bins=100,
             label="Gammas")
    plt.hist(dataforwl[dataforwl[hadro] > 0]['width'],
             histtype=u'step', bins=100,
             label="Protons")
    plt.ylabel(r'# of events', fontsize=15)
    plt.xlabel(r"Width (º)")

    # Length distribution
    plt.subplot(335)
    plt.hist(dataforwl[dataforwl[hadro] < 1]['length'],
             histtype=u'step', bins=100,
             label="Gammas")
    plt.hist(dataforwl[dataforwl[hadro] > 0]['length'],
             histtype=u'step', bins=100,
             label="Protons")
    plt.ylabel(r'# of events', fontsize=15)
    plt.xlabel(r"Length (º)")

    # r distribution
    plt.subplot(336)
    plt.hist(data[data[hadro] < 1]['r'],
             histtype=u'step', bins=100,
             label="Gammas")
    plt.hist(data[data[hadro] > 0]['r'],
             histtype=u'step', bins=100,
             label="Protons")
    plt.ylabel(r'# of events', fontsize=15)
    plt.xlabel(r"r (m)")

    # psi distribution

    plt.subplot(337)
    plt.hist(data[data[hadro] < 1]['psi'],
             histtype=u'step', bins=100,
             label="Gammas")
    plt.hist(data[data[hadro] > 0]['psi'],
             histtype=u'step', bins=100,
             label="Protons")
    plt.ylabel(r'# of events', fontsize=15)
    plt.xlabel(r"psi angle(rad)")

    # psi distribution

    plt.subplot(338)
    plt.hist(data[data[hadro] < 1]['phi'],
             histtype=u'step', bins=100,
             label="Gammas")
    plt.hist(data[data[hadro] > 0]['phi'],
             histtype=u'step', bins=100,
             label="Protons")
    plt.ylabel(r'# of events', fontsize=15)
    plt.xlabel(r"phi angle(m)")

    # Time gradient

    plt.subplot(339)
    plt.hist(data[data[hadro] < 1]['time_gradient'],
             histtype=u'step', bins=100,
             label="Gammas")
    plt.hist(data[data[hadro] > 0]['time_gradient'],
             histtype=u'step', bins=100,
             label="Protons")
    plt.ylabel(r'# of events', fontsize=15)
    plt.xlabel(r"Time gradient")


def energy_results(dl2_data, points_outfile=None, plot_outfile=None):
    """
    Plot energy resolution, energy bias and energy migration matrix in the same figure

    Parameters
    ----------
    dl2_data: `pandas.DataFrame`
        dl2 MC gamma data - must include the columns `mc_energy` and `reco_energy`
    points_outfile: None or str
        if specified, save the resolution and bias in hdf5 format
    plot_outfile: None or str
        if specified, save the figure

    Returns
    -------
    fig, axes: `matplotlib.pyplot.figure`, `matplotlib.pyplot.axes`
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ctaplot.resolution_per_energy(dl2_data.mc_energy, dl2_data.reco_energy, dl2_data.reco_energy)
    ctaplot.plot_energy_resolution(dl2_data.mc_energy, dl2_data.reco_energy, ax=axes[0, 0], bias_correction=False)
    ctaplot.plot_energy_resolution_cta_requirement('north', ax=axes[0, 0], color='black')
    ctaplot.plot_energy_bias(dl2_data.mc_energy, dl2_data.reco_energy, ax=axes[1, 0])
    ctaplot.plot_migration_matrix(dl2_data.mc_energy.apply(np.log10),
                                  dl2_data.reco_energy.apply(np.log10),
                                  ax=axes[0, 1],
                                  colorbar=True,
                                  xy_line=True,
                                  hist2d_args=dict(norm=matplotlib.colors.LogNorm()),
                                  line_args=dict(color='black'),
                                  )
    axes[0, 0].legend()
    axes[0, 1].set_xlabel('log(mc energy/[TeV])')
    axes[0, 1].set_ylabel('log(reco energy/[TeV])')
    axes[0, 0].set_title("")
    axes[0, 0].label_outer()
    axes[1, 0].set_title("")
    axes[1, 0].set_ylabel("Energy bias")
    for ax in axes.ravel(): ax.grid(which='both')
    axes[1, 1].remove()

    fig.tight_layout()

    if points_outfile:
        e_bins, e_res = ctaplot.energy_resolution_per_energy(dl2_data.mc_energy, dl2_data.reco_energy)
        e_bins, e_bias = ctaplot.energy_bias(dl2_data.mc_energy, dl2_data.reco_energy)
        write_energy_resolutions(points_outfile, e_bins * u.TeV, e_res, e_bias)

    if plot_outfile:
        fig.savefig(plot_outfile)

    return fig, axes


def write_energy_resolutions(outfile, e_bins, res, bias=None, overwrite=False, append=True):
    """
    Save the computed resolutions in hdf5 format

    Parameters
    ----------
    outfile: str
    e_bins: `numpy.ndarray`
    res: `np.ndarray`
    bias: `np.ndarray`
    overwrite
    append
    """
    e_bins_t = Table(data=e_bins[..., np.newaxis], names=['energy_bins'])

    data = res
    names = ['energy_res', 'energy_res_err_lo', 'energy_res_err_hi']

    if bias is not None:
        data = np.append(data, bias[..., np.newaxis], axis=1)
        names.append('energy_bias')

    res_t = Table(data=data, names=names)
    write_table_hdf5(e_bins_t, outfile, path='bins', overwrite=overwrite, append=append, serialize_meta=True)
    write_table_hdf5(res_t, outfile, path='res', append=True)


def write_angular_resolutions(outfile, e_bins, res, overwrite=False, append=True):
    """

    Parameters
    ----------
    outfile: str
    e_bins: `numpy.ndarray`
    res: `np.ndarray`
    overwrite
    append
    """
    e_bins_t = Table(data=e_bins[..., np.newaxis], names=['energy_bins'])

    data = res
    names = ['angular_res', 'angular_res_err_lo', 'angular_res_err_hi']

    res_t = Table(data=data, names=names)
    write_table_hdf5(e_bins_t, outfile, path='bins', overwrite=overwrite, append=append, serialize_meta=True)
    write_table_hdf5(res_t, outfile, path='res', append=True)


def read_resolutions(filename):
    """
    Read resolutions from hdf5 file

    Parameters
    ----------
    filename: str

    Returns
    -------
    bins, res: `astropy.table.Table, astropy.table.Table`
    """
    bins = read_table_hdf5(filename, path='bins')
    res = read_table_hdf5(filename, path='res')
    return bins, res


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

    reco_disp_norm = np.sqrt(gammas['reco_disp_dx'] ** 2 + gammas['reco_disp_dy'] ** 2)
    disp_res = ((gammas['disp_norm'] - reco_disp_norm) / gammas['disp_norm'])

    section = disp_res[abs(disp_res) < 0.5]
    mu, sigma = norm.fit(section)
    print("mu = {}\n sigma = {}".format(mu, sigma))

    n, bins, patches = plt.hist(disp_res,
                                bins=100,
                                density=1,
                                alpha=0.75,
                                range=[-2, 1.5],
                                )

    y = norm.pdf(bins, mu, sigma)

    plt.plot(bins, y, 'r--', linewidth=2)

    plt.xlabel(r'$\\frac{disp\_norm_{gammas}-disp_{rec}}{disp\_norm_{gammas}}$', fontsize=15)

    plt.figtext(0.15, 0.7, 'Mean: ' + str(round(mu, 4)), fontsize=12)
    plt.figtext(0.15, 0.65, 'Std: ' + str(round(sigma, 4)), fontsize=12)

    plt.subplot(222)

    hD = plt.hist2d(gammas['disp_norm'], reco_disp_norm,
                    bins=100,
                    range=([0, 1.1], [0, 1.1]),
                    )

    plt.colorbar(hD[3])
    plt.xlabel(r'$disp\_norm_{gammas}$', fontsize=15)

    plt.ylabel(r'$disp\_norm_{rec}$', fontsize=15)

    plt.plot(gammas['disp_norm'], gammas['disp_norm'], "-", color='red')

    plt.subplot(223)
    theta2 = (gammas['src_x'] - gammas['reco_src_x']) ** 2 + (gammas['src_y'] - gammas['src_y']) ** 2

    plt.hist(theta2, bins=100, range=[0, 0.1], histtype=u'step')
    plt.xlabel(r'$\theta^{2}(º)$', fontsize=15)
    plt.ylabel(r'# of events', fontsize=15)


def plot_disp_vector(data):
    fig, axes = plt.subplots(1, 2)

    axes[0].hist2d(data.disp_dx, data.reco_disp_dx, bins=60)
    axes[0].set_xlabel('mc_disp')
    axes[0].set_ylabel('reco_disp')
    axes[0].set_title('disp_dx')

    axes[1].hist2d(data.disp_dy, data.reco_disp_dy, bins=60)
    axes[1].set_xlabel('mc_disp')
    axes[1].set_ylabel('reco_disp')
    axes[1].set_title('disp_dy')


def plot_pos(data, true_hadroness=False):
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

    # True position

    trueX = data[data[hadro] == 0]['src_x']
    trueY = data[data[hadro] == 0]['src_y']
    trueXprot = data[data[hadro] == 101]['src_x']
    trueYprot = data[data[hadro] == 101]['src_y']

    # Reconstructed position

    recX = data[data[hadro] == 0]['reco_src_x']
    recY = data[data[hadro] == 0]['reco_src_y']
    recXprot = data[data[hadro] == 101]['reco_src_x']
    recYprot = data[data[hadro] == 101]['reco_src_y']
    ran = np.array([(-0.3, 0.3), (-0.4, 0.4)])
    nbins = 50

    plt.subplot(221)
    plt.hist2d(trueXprot, trueYprot,
               bins=nbins,
               label="Protons",
               range=ran)
    plt.colorbar()
    plt.title("True position Protons")
    plt.xlabel("x(m)")
    plt.ylabel("y (m)")

    plt.subplot(222)
    plt.hist2d(trueX, trueY,
               bins=nbins,
               label="Gammas",
               range=ran)
    plt.colorbar()
    plt.title("True position Gammas")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    plt.subplot(223)
    plt.hist2d(recXprot, recYprot,
               bins=nbins,
               label="Protons",
               range=ran)
    plt.colorbar()
    plt.title("Reconstructed position Protons")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    plt.subplot(224)
    plt.hist2d(recX, recY,
               bins=nbins,
               label="Gammas",
               range=ran,
               )
    plt.colorbar()
    plt.title("Reconstructed position Gammas")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")


def plot_importances(model, features_names, ax=None, **kwargs):
    """
    plot features importances
    
    Parameters
    ----------
    model: scikit-learn model
    features_names: list
    ax: `matplotlib.pyplot.axes`
    kwargs: kwargs for `matplot.pyplot.barh`

    Returns
    -------
    ax: `matplotlib.pyplot.axis`
    """

    ax = plt.gca() if ax is None else ax

    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)

    ordered_features = []
    for index in indices:
        ordered_features = ordered_features + [features_names[index]]

    ax.set_title("Feature importances (gini index)")

    ax.barh(range(len(features_names)),
            importances[indices],
            xerr=std[indices],
            align="center",
            **kwargs
            )

    ax.set_yticks(range(len(features_names)))
    ax.set_yticklabels(np.array(features_names)[indices])
    ax.grid()

    return ax


def plot_models_features_importances(path_models, config_file=None, axes=None, **kwargs):
    """
    Plot features importances for the trained models

    Parameters
    ----------
    path_models: path the trained models
    config_file: None or str
        Path to the configuration file used to train the models
        If None is provided, it is assumed that the standard configuration has been used
    axes: None or list of `matplotlib.pyplot.axes` objects
        If None, a figure with 3 subplots is created
    kwargs: args for `matplotlib.pyplot.barh`

    Returns
    -------
    axes: list of `matplotlib.pyplot.axes` objects
    """

    if config_file is None:
        config = get_standard_config()
    else:
        config = read_configuration_file(config_file)

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    else:
        fig = axes[0].get_figure()

    fig.suptitle('Features importances')

    ### Regression models ###
    reg_features_names = config['regression_features']

    energy = joblib.load(os.path.join(path_models, "reg_energy.sav"))
    disp = joblib.load(os.path.join(path_models, "reg_disp_vector.sav"))

    plot_importances(disp, reg_features_names, ax=axes[0], **kwargs)
    axes[0].set_title("disp")

    plot_importances(energy, reg_features_names, ax=axes[1], **kwargs)
    axes[1].set_title("energy")

    ### Classification model ###
    clf_features_names = config['classification_features']
    clf = joblib.load(os.path.join(path_models, "cls_gh.sav"))

    plot_importances(clf, clf_features_names, ax=axes[2], **kwargs)
    axes[2].set_title("classification")

    fig.tight_layout()

    return axes


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
    Plot the energy resolution from a pandas dataframe of DL2 data.
    See `~ctaplot.plot_energy_resolution` for doc.

    Parameters
    ----------
   dl2_data: `pandas.DataFrame`
        Reconstructed MC events at DL2+ level.
    ax: `matplotlib.pyplot.axes` or None
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
    ax.grid(which='both')
    if cta_req_north:
        ax = ctaplot.plot_energy_resolution_cta_requirement('north', ax=ax, color='black')
    return ax


def plot_angular_resolution(dl2_data, ax=None, bias_correction=False, cta_req_north=False, **kwargs):
    """
    Plot the energy resolution from a pandas dataframe of DL2 data.
    See `~ctaplot.plot_energy_resolution` for doc.

    Parameters
    ----------
    dl2_data: `pandas.DataFrame`
        Reconstructed MC events at DL2+ level.
    ax: `matplotlib.pyplot.axes` or None
    bias_correction: `bool`
        correct for systematic bias
    cta_req_north: `bool`
        if True, includes CTA requirement curve
    kwargs: args for `matplotlib.pyplot.plot`

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    ax = ctaplot.plot_angular_resolution_per_energy(dl2_data.reco_alt,
                                                    dl2_data.reco_az,
                                                    dl2_data.mc_alt,
                                                    dl2_data.mc_az,
                                                    dl2_data.reco_energy,
                                                    ax=ax,
                                                    bias_correction=bias_correction,
                                                    **kwargs
                                                    )
    ax.grid(which='both')
    if cta_req_north:
        ax = ctaplot.plot_angular_resolution_cta_requirement('north', ax=ax, color='black')

    return ax


def direction_results(dl2_data, points_outfile=None, plot_outfile=None):
    """
    
    Parameters
    ----------
    dl2_data: `pandas.DataFrame`
    points_outfile: None or str
        filename to save angular resolution data points
    plot_outfile: None or str
        filename to save the figure

    Returns
    -------
    fig, axes: `matplotlib.pyplot.figure`, `matplotlib.pyplot.axes`
    """

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    ax = ctaplot.plot_theta2(dl2_data.reco_alt,
                             dl2_data.reco_az,
                             dl2_data.mc_alt,
                             dl2_data.mc_az,
                             ax=axes[0, 0],
                             bins=100,
                             range=(0, 1),
                             )
    ax.grid()

    ctaplot.plot_angular_resolution_per_energy(dl2_data.reco_alt,
                                               dl2_data.reco_az,
                                               dl2_data.mc_alt,
                                               dl2_data.mc_az,
                                               dl2_data.reco_energy,
                                               ax=axes[0, 1],
                                               )

    ctaplot.plot_angular_resolution_cta_requirement('north', ax=axes[0, 1], color='black')
    axes[0, 1].grid()
    axes[0, 1].legend()

    ctaplot.plot_migration_matrix(dl2_data.mc_alt,
                                  dl2_data.reco_alt,
                                  ax=axes[1, 0],
                                  colorbar=True,
                                  xy_line=True,
                                  hist2d_args=dict(norm=matplotlib.colors.LogNorm()),
                                  line_args=dict(color='black'),
                                  )
    axes[1, 0].set_xlabel('simu alt [rad]')
    axes[1, 0].set_ylabel('reco alt [rad]')

    ctaplot.plot_migration_matrix(dl2_data.mc_az,
                                  dl2_data.reco_az,
                                  ax=axes[1, 1],
                                  colorbar=True,
                                  xy_line=True,
                                  hist2d_args=dict(norm=matplotlib.colors.LogNorm()),
                                  line_args=dict(color='black'),
                                  )
    axes[1, 1].set_xlabel('simu az [rad]')
    axes[1, 1].set_ylabel('reco az [rad]')

    fig.tight_layout()

    if points_outfile:
        e_bins, ang_res = ctaplot.angular_resolution_per_energy(dl2_data.reco_alt,
                                                                dl2_data.reco_az,
                                                                dl2_data.mc_alt,
                                                                dl2_data.mc_az,
                                                                dl2_data.reco_energy,
                                                                )

        write_angular_resolutions(points_outfile, e_bins * u.TeV, ang_res * u.rad)

    if plot_outfile:
        fig.savefig(plot_outfile)

    return fig, axes


def plot_wobble(source_position, n_points, ax=None):
    """
    Plot 2D map of ON/OFF positions w.r.t. to the camera center

    Parameters
    ----------
    source_position: Source position in the camera frame, array-like [x,y]
    n_points: Number of observation points. Rotation angle for each next observation is determined
    as 360/n_points
    ax: `matplotlib.pyplot.axes` or None

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """
    from lstchain.reco.utils import rotate
    if ax is None:
        ax = plt.gca()
    opacity = 0.2
    marker_size = 20
    color_map_name = 'Set1'  # https://matplotlib.org/gallery/color/colormap_reference.html
    colors = get_cmap(color_map_name).colors
    ax.set_prop_cycle(color=colors)

    rotation_angle = 360. / n_points
    labels = ['Source', ] + [f'OFF {rotation_angle * x}' for x in range(1, n_points)]
    ax.plot((0, 0), '.', markersize=marker_size, alpha=opacity, color='black', label="Camera center")
    for off_point in range(n_points):
        first_point = tuple(rotate(list(zip(source_position[0].to_value(),
                                            source_position[1].to_value()))[0],
                                   rotation_angle * off_point)[0])
        ax.plot(first_point[0], first_point[1], '.', markersize=marker_size, alpha=opacity,
                label=labels[off_point])
        ax.annotate(labels[off_point], xy=(first_point[0] - 0.1, first_point[1] + 0.05), label=labels[off_point])

    ax.set_ylim(-0.7, 0.7)
    ax.set_xlim(-0.7, 0.7)

    ax.set_ylabel("(m)")
    ax.set_xlabel("Position in the camera (m)")
    return ax


def plot_1d_excess(named_datasets, lima_significance,
                   x_label, x_cut, ax=None, x_range_min=0, x_range_max=2,
                   n_bins=100, opacity=0.2, color_map_name='Set1'):
    """
    Plot one-dimensional distribution of signal and backgound events
    Color maps: https://matplotlib.org/gallery/color/colormap_reference.html

    Parameters
    ----------
    named_datasets: Array of datasets to plot in a following form: (<dataset label>, data, overall
    scale factor)
    lima_significance: Li&Ma significance of observation
    x_label: X-axis label
    x_cut: X cut value
    ax: `matplotlib.pyplot.axes` or None
    x_range_min: Bottom value of X
    x_range_max: Top value of X
    n_bins: Number of histogram bins along X axis
    opacity: Plot opaacity
    color_map_name: Matplotlib colormap name

    Returns
    -------
    ax: `matplotlib.pyplot.axes`
    """

    if ax is None:
        ax = plt.gca()
    colors = get_cmap(color_map_name).colors
    ax.set_prop_cycle(color=colors)

    hists = []
    for label, data, factor in named_datasets:
        hists.append(ax.hist(data, label=label, weights=factor * np.ones_like(data),
                             bins=n_bins, alpha=opacity, range=[x_range_min, x_range_max]))

    ax.annotate(text=rf'Significance Li&Ma = {lima_significance:.2f} $\sigma$\n',
                xy=(np.max(hists[0][1] / 4), np.max(hists[0][0] / 6 * 5)), size=20, color='r')

    ax.vlines(x=x_cut, ymin=0, ymax=np.max(hists[0][0] * 1.2), linestyle='--', linewidth=2,
              color='black', alpha=opacity)
    ax.set_xlabel(x_label)
    ax.set_ylabel(r'Number of events')
    ax.legend(fontsize=12)
    return ax
