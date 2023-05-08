import numpy as np
import matplotlib.pyplot as plt
from copy import copy
import astropy.units as u
from ctapipe.visualization import CameraDisplay
from lstchain.reco.reconstructorCC import asygaussian2d as asygaussian2d

__all__ = [
    'plot_1dlikelihood',
    'plot_2dlikelihood',
    'plot_debug',
    'plot_event',
    'plot_likelihood',
    'plot_model',
    'plot_residual',
    'plot_waveforms',
]

labels = {'charge': 'Charge [p.e.]',
          't_cm': '$t_{CM}$ [ns]',
          'x_cm': '$x_{CM}$ [m]',
          'y_cm': '$y_{CM}$ [m]',
          'length': r'$\sigma_l$ [m]',
          'wl': r'$\sigma_w$ / $\sigma_l$',
          'psi': r'$\psi$ [rad]',
          'v': '$v$ [m/ns]',
          'rl': 'length asymmetry'
          }


def plot_debug(fitter, event, telescope_id, dl1_container, identifier):
    """
    Create a set of plots for one event

    """
    image = event.dl1.tel[telescope_id].image
    geometry = fitter.subarray.tel[telescope_id].camera.geometry
    clean_mask = event.dl1.tel[telescope_id].image_mask
    plot_event(fitter, image, geometry, init=True, clean_mask=clean_mask, save=True, ids=identifier)
    plot_waveforms(fitter, event, telescope_id, save=True, ids=identifier)
    plot_event(fitter, image, geometry, save=True, ids=identifier)
    plot_residual(fitter, image, geometry, save=True, ids=identifier)
    plot_model(fitter, geometry, save=True, ids=identifier)
    focal_length = fitter.subarray.tel[telescope_id].optics.equivalent_focal_length
    angle_dist_eq = [(u.rad, u.m, lambda x: np.tan(x) * focal_length.to_value(u.m),
                      lambda x: np.arctan(x / focal_length.to_value(u.m))),
                     (u.rad**2, u.m**2, lambda x: (np.tan(np.sqrt(x)) * focal_length.to_value(u.m))**2,
                      lambda x: (np.arctan(np.sqrt(x) / focal_length.to_value(u.m)))**2)]
    with u.set_enabled_equivalencies(angle_dist_eq):
        _, fit_params = fitter.call_setup(event, telescope_id, dl1_container)
        for params in fitter.start_parameters.keys():
            plot_likelihood(fitter, fit_params, params, save=True, ids=identifier)
    plot_likelihood(fitter, fit_params, 'x_cm', 'y_cm', save=True, ids=identifier)

    if fitter.verbose == 3:
        print("event plot produced, press Enter to continue or Ctrl+C and Enter to stop")
        input()


def plot_1dlikelihood(fitter, fit_params, parameter_name, axes=None, size=1000,
                      x_label=None, invert=False, loc='best'):
    """
        Plot the 1D evolution of the log-likelihood for a parameter
        when fixing the other parameters to their end value.

        Parameters
        ----------
        parameter_name: string
            Parameter over which the log-likelihood needs to be plotted
        axes: matplotlib.pyplot.axis
            Axis used to store the figure
            If None, a new one is created
        size: int
            Number of points of the likelihood curve
        x_label: string
            Label of the x axis
        invert: bool
            If True, invert the x and y axis
        loc: string
            Legend position

        Returns
        -------
        axes: matplotlib.pyplot.axis
            Axis object filled with the 1D log-likelihood figure

    """
    key = parameter_name

    if key not in fitter.names_parameters:
        raise NameError('Parameter : {} not in existing parameters :'
                        '{}'.format(key, fitter.names_parameters))

    x = np.linspace(fitter.bound_parameters[key][0],
                    fitter.bound_parameters[key][1], num=size)
    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(111)

    params = copy(fitter.end_parameters)
    llh = np.zeros(x.shape)

    for i, xx in enumerate(x):
        params[key] = xx
        try:
            y = fitter.log_likelihood(*params.values(), fit_params=fit_params)
        except ZeroDivisionError:
            pass
        else:
            llh[i] = y

    x_label = labels[key] if x_label is None else x_label

    if not invert:
        axes.plot(x, -llh, color='r')
        axes.axvline(fitter.end_parameters[key], linestyle='--', color='k',
                     label='Fitted value {:.2f}'.format(
                         fitter.end_parameters[key]))
        axes.axvline(fitter.start_parameters[key], linestyle='--',
                     color='b', label='Starting value {:.2f}'.format(
                fitter.start_parameters[key]
            ))
        axes.set_ylabel(r'-$\ln \mathcal{L}$')
        axes.set_xlabel(x_label)

    else:

        axes.plot(-llh, x, color='r')
        axes.axhline(fitter.end_parameters[key], linestyle='--',
                     color='k',
                     label='Fitted value {:.2f}'.format(
                         fitter.end_parameters[key]))
        axes.axhline(fitter.start_parameters[key], linestyle='--',
                     color='b', label='Starting value {:.2f}'.format(
                fitter.start_parameters[key]
            ))
        axes.axhspan(fitter.bound_parameters[key][0],
                     fitter.bound_parameters[key][1], label='bounds',
                     alpha=0.5, facecolor='k')
        axes.set_xlabel(r'-$\ln \mathcal{L}$')
        axes.set_ylabel(x_label)
        axes.xaxis.set_label_position('top')

    axes.legend(loc=loc)
    return axes


def plot_2dlikelihood(fitter, fit_params, parameter_1, parameter_2=None, size=100,
                      x_label=None, y_label=None):
    """
        Plot the 2D evolution of the log-likelihood for a pair of
        parameters when fixing the other parameters to their end value.

        Parameters
        ----------
        parameter_1: string
            First parameter over which the function needs to be plotted
        parameter_2: string
            Second parameter over which the function needs to be plotted
        size: int or (int, int)
            Number of points of the likelihood per dimension
        x_label: string
            Label of the x axis
        y_label: string
            Label of the y axis

        Returns
        -------
        axes: matplotlib.pyplot.axis
            Axis object filled with the 2D log-likelihood figures

    """

    if isinstance(size, int):
        size = (size, size)

    key_x = parameter_1
    key_y = parameter_2
    x = np.linspace(fitter.bound_parameters[key_x][0],
                    fitter.bound_parameters[key_x][1], num=size[0])
    y = np.linspace(fitter.bound_parameters[key_y][0],
                    fitter.bound_parameters[key_y][1], num=size[1])
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    params = copy(fitter.end_parameters)
    llh = np.zeros(size)

    for i, xx in enumerate(x):
        params[key_x] = xx
        for j, yy in enumerate(y):
            params[key_y] = yy
            llh[i, j] = fitter.log_likelihood(*params.values(), fit_params=fit_params)

    fig = plt.figure()
    left, width = 0.1, 0.6
    bottom, height = 0.1, 0.6
    spacing = 0.005
    rect_center = [left, bottom, width, height]
    rect_x = [left, bottom + height + spacing, width, 0.2]
    rect_y = [left + width + spacing, bottom, 0.2, height]
    axes = fig.add_axes(rect_center)
    axes_x = fig.add_axes(rect_x)
    axes_y = fig.add_axes(rect_y)
    axes.tick_params(direction='in', top=True, right=True)
    plot_1dlikelihood(fitter, fit_params, parameter_name=parameter_1, axes=axes_x,
                      loc='upper left')
    plot_1dlikelihood(fitter, fit_params, parameter_name=parameter_2, axes=axes_y,
                      invert=True, loc='lower right')
    axes_x.tick_params(direction='in', labelbottom=False)
    axes_y.tick_params(direction='in', labelleft=False)

    axes_x.set_xlabel('')
    axes_y.set_ylabel('')
    axes_x.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)
    axes_y.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)

    x_label = labels[key_x] if x_label is None else x_label
    y_label = labels[key_y] if y_label is None else y_label

    im = axes.imshow(-llh.T, origin='lower', extent=[x.min() - dx / 2.,
                                                     x.max() - dx / 2.,
                                                     y.min() - dy / 2.,
                                                     y.max() - dy / 2.],
                     aspect='auto')

    axes.scatter(fitter.end_parameters[key_x], fitter.end_parameters[key_y],
                 marker='x', color='w', label='- log Likelihood')
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.legend(loc='upper left')
    plt.colorbar(mappable=im, ax=axes_y, label=r'-$\ln \mathcal{L}$')
    axes_x.set_xlim(x.min(), x.max())
    axes_y.set_ylim(y.min(), y.max())

    return axes


def plot_likelihood(fitter, fit_params, parameter_1, parameter_2=None,
                    axes=None, size=100,
                    x_label=None, y_label=None,
                    save=False, ids=''):
    """
        Plot the 1D or 2D likelihood.

        Parameters
        ----------
        parameter_1: string
            Parameter over which the function needs to be plotted
        parameter_2: string
            Second parameter over which the function needs to be plotted
        axes: `matplotlib.pyplot.axis`
            Axis used to store the figure
            If None, a new one is created
        size: int
            Number of points of the likelihood curve for each dimension
        x_label: string
            Label of the x axis
        y_label: string
            Label of the y axis
        save: bool
            Save and close the figure if True, return it otherwise
        ids: string
            Can be used to modify the save location

        Returns
        -------
        None or `matplotlib.pyplot.axis` object filled with the log-likelihood figure

    """
    if parameter_2 is None:
        axes = plot_1dlikelihood(fitter, fit_params, parameter_name=parameter_1,
                                 axes=axes, x_label=x_label, size=size)
    else:
        axes = plot_2dlikelihood(fitter, fit_params, parameter_1, parameter_2=parameter_2,
                                 size=size, x_label=x_label, y_label=y_label)
    if save:
        axes.get_figure().savefig('event/' + ids + '_'
                                  + parameter_1 + '_' + str(parameter_2) + '.png')
        plt.close()
    return None if save else axes


def plot_event(fitter, image, geometry, n_sigma=3, init=False, clean_mask=None, show_ellipsis=True, save=False, ids=''):
    """
        Plot the image of the event in the camera along with the extracted
        ellipsis before or after the fitting procedure.

    Parameters
    ----------
    image:
        Distribution of signal for the event in number of p.e.
    n_sigma: float
        Multiplicative factor on the extracted width and length
        used for the displayed ellipsis
    init: boolean
        If True, use the starting parameters for the ellipsis
        If False, use the ending parameters for the ellipsis
    clean_mask: boolean array
        cleaning selected pixels for the Hillas parameters extraction
    show_ellipsis: boolean
        If True, display the ellipsis
    save: bool
        Save and close the figure if True, return it otherwise
    ids: string
        Can be used to modify the save location
    Returns
    -------
    cam_display: `ctapipe.visualization.CameraDisplay`
        Camera image using matplotlib

    """

    fig, axes = plt.subplots(figsize=(10, 8))
    cam_display = CameraDisplay(geometry, image, ax=axes)
    cam_display.add_colorbar(ax=axes)
    if init:
        params = fitter.start_parameters
    else:
        params = fitter.end_parameters

    length = n_sigma * params['length']
    psi = params['psi']
    if show_ellipsis:
        cam_display.add_ellipse(centroid=(params['x_cm'],
                                          params['y_cm']),
                                width=n_sigma * params['wl'] * params['length'],
                                length=length,
                                angle=psi,
                                linewidth=6, color='r', linestyle='--',
                                label=r'{} $\sigma$ contour'.format(n_sigma))
        cam_display.axes.legend(loc='best')

    if init and clean_mask is not None:
        cam_display.highlight_pixels(clean_mask, color='r')

    if save:
        cam_display.axes.get_figure().savefig('event/' + ids +
                                              '_init' + str(init) + '.png')
        plt.close()
    return None if save else cam_display


def plot_residual(fitter, image, geometry, save=False, ids=''):
    """
        Plot the residuals image- spatial_model in the camera after fitting

    Parameters
    ----------
    image:
        Distribution of signal for the event in number of p.e.
    save: bool
        Save and close the figure if True, return it otherwise
    ids: string
        Can be used to modify the save location

    Returns
    -------
    cam_display: `ctapipe.visualization.CameraDisplay`
        Camera image using matplotlib

    """

    params = fitter.end_parameters

    rl = 1 + params['rl'] if params['rl'] >= 0 else 1 / (1 - params['rl'])
    mu = asygaussian2d(params['charge'] * geometry.pix_area.to_value(u.m ** 2),
                       geometry.pix_x.value,
                       geometry.pix_y.value,
                       params['x_cm'],
                       params['y_cm'],
                       params['wl'] * params['length'],
                       params['length'],
                       params['psi'],
                       rl)
    residual = image - mu

    fig, axes = plt.subplots(figsize=(10, 8))
    cam_display = CameraDisplay(geometry, residual, ax=axes)
    cam_display.add_colorbar(ax=axes)
    if save:
        cam_display.axes.get_figure().savefig('event/' + ids +
                                              '_residuals.png')
        plt.close()
    return None if save else cam_display


def plot_model(fitter, geometry, save=False, ids=''):
    """
    Create a CameraDisplay object showing the spatial model fitted to
    the current event

    Parameters
    -------
    save: bool
        Save and close the figure if True, return it otherwise
    ids: string
        Can be used to modify the save location

    Returns
    -------
    cam_display: `ctapipe.visualization.CameraDisplay`
        Camera image using matplotlib

    """

    params = fitter.end_parameters
    rl = 1 + params['rl'] if params['rl'] >= 0 else 1 / (1 - params['rl'])
    mu = asygaussian2d(params['charge'] * geometry.pix_area.to_value(u.m ** 2),
                       geometry.pix_x.value,
                       geometry.pix_y.value,
                       params['x_cm'],
                       params['y_cm'],
                       params['wl'] * params['length'],
                       params['length'],
                       params['psi'],
                       rl)

    fig, axes = plt.subplots(figsize=(10, 8))
    cam_display = CameraDisplay(geometry, mu, ax=axes)
    cam_display.add_colorbar(ax=axes)

    if save:
        cam_display.axes.get_figure().savefig('event/' + ids +
                                              '_model.png')
        plt.close()
    return None if save else cam_display


def plot_waveforms(fitter, event, telescope_id, axes=None, save=False, ids=''):
    """
        Plot the intensity of the signal in the camera as a function of
        time and of the position projected on the main axis of the fitted
        ellipsis.

    Parameters
    ----------
    axes: `matplotlib.pyplot.axis`
        Axis used to store the figure
        If None, a new one is created
    save: bool
        Save and close the figure if True, return it otherwise
    ids: string
        Can be used to modify the save location

    Returns
    -------
    axes: `matplotlib.pyplot.axis`
        Object filled with the figure
    """
    image = event.dl1.tel[telescope_id].image
    geometry = fitter.subarray.tel[telescope_id].camera.geometry
    data = event.r1.tel[telescope_id].waveform
    n_pixels = min(20, len(image))
    pixels = np.argsort(image)[-n_pixels:]
    dx = (geometry.pix_x[pixels].to_value() - fitter.end_parameters['x_cm'])
    dy = (geometry.pix_y[pixels].to_value() - fitter.end_parameters['y_cm'])
    long_pix = dx * np.cos(fitter.end_parameters['psi']) + dy * np.sin(
        fitter.end_parameters['psi'])
    fitted_times = np.polyval(
        [fitter.end_parameters['v'], fitter.end_parameters['t_cm']], long_pix)
    times_index = np.argsort(fitted_times)

    waveforms = data[pixels]
    waveforms = waveforms[times_index]
    long_pix = long_pix[times_index]
    fitted_times = fitted_times[times_index]
    n_pixels, n_samples = waveforms.shape
    times = np.arange(0, n_samples)
    X, Y = np.meshgrid(times, long_pix)

    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(111)
    M = axes.pcolormesh(X, Y, waveforms)
    axes.set_xlabel('time [ns]')
    axes.set_ylabel('Longitude [m]')
    label = (labels['t_cm']
             + ' : {:.2f} [ns]'.format(fitter.end_parameters['t_cm']))
    label += ('\n' + labels['v']
              + ' : {:.2f} [m/ns]'.format(fitter.end_parameters['v']))
    axes.plot(fitted_times, long_pix, color='r',
              label=label)
    axes.legend(loc='best')
    axes.get_figure().colorbar(label='[p.e.]', ax=axes, mappable=M)

    if save:
        axes.get_figure().savefig('event/' + ids +
                                  '_waveform.png')
        plt.close()
    return None if save else axes
