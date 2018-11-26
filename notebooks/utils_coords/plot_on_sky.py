import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.axisartist import SubplotHost, ParasiteAxesAuxTrans
import mpl_toolkits.axisartist.angle_helper as angle_helper
from mpl_toolkits.axisartist.grid_helper_curvelinear import GridHelperCurveLinear

from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D


def sky_in_box(fig, az_deg, zen_deg, label, center):
    """
    polar projection, but in a rectangular box.
    :param fig:
    :param az_deg:
    :param zen_deg:
    :param label:
    :param center: array pointing to have the center of the image
    :return:
    """
    # PolarAxes.PolarTransform takes radian. However, we want our coordinate
    # system in degree
    tr = Affine2D().scale(np.pi/180., 1.).translate(-np.pi/2.,0) + PolarAxes.PolarTransform()
    # polar projection, which involves cycle, and also has limits in
    # its coordinates, needs a special method to find the extremes
    # (min, max of the coordinate within the view).

    # 20, 20 : number of sampling points along x, y direction
    extreme_finder = angle_helper.ExtremeFinderCycle(20, 20,
                                                     lon_cycle=360,
                                                     lat_cycle=None,
                                                     lon_minmax=None,
                                                     #lat_minmax=(0, np.inf),
                                                     lat_minmax=(-90, 90),
                                                     )

    grid_locator1 = angle_helper.LocatorDMS(10)
    # Find a grid values appropriate for the coordinate (degree,
    # minute, second).

    tick_formatter1 = angle_helper.FormatterDMS()
    # And also uses an appropriate formatter.  Note that,the
    # acceptable Locator and Formatter class is a bit different than
    # that of mpl's, and you cannot directly use mpl's Locator and
    # Formatter here (but may be possible in the future).

    grid_helper = GridHelperCurveLinear(tr,
                                        extreme_finder=extreme_finder,
                                        grid_locator1=grid_locator1,
                                        tick_formatter1=tick_formatter1
                                        )

    ax1 = SubplotHost(fig, 1, 1, 1, grid_helper=grid_helper)

    # make ticklabels of right and top axis visible.
    ax1.axis["right"].major_ticklabels.set_visible(True)
    ax1.axis["top"].major_ticklabels.set_visible(True)

    # let right axis shows ticklabels for 1st coordinate (angle)
    ax1.axis["right"].get_helper().nth_coord_ticks = 0
    # let bottom axis shows ticklabels for 2nd coordinate (radius)
    ax1.axis["bottom"].get_helper().nth_coord_ticks = 1

    fig.add_subplot(ax1)

    # A parasite axes with given transform
    ax2 = ParasiteAxesAuxTrans(ax1, tr, "equal")
    # note that ax2.transData == tr + ax1.transData
    # Anything you draw in ax2 will match the ticks and grids of ax1.
    ax1.parasites.append(ax2)
    #intp = cbook.simple_linear_interpolation
    # ax2.scatter(intp(np.array([0, 0]), 10),
    #          intp(np.array([10., 10.]), 10),
    #          linewidth=2.0)

    for i in range(len(az_deg)):
        ax2.scatter(az_deg[i], zen_deg[i], label=label[i])

    ax1.set_aspect(1.)
    ax1.set_xlim(-2.5, 2.5)

    # TODO: improve this part. make it more general
    if np.abs(center.az.value-180) < 3:
        ax1.set_ylim(17.5, 22.5)
    elif center.az.value < 3:
        ax1.set_ylim(-22.5, -17.5)

    ax1.grid(True, zorder=0)

    return ax1


if __name__ == '__main__':

    fig = plt.figure(1, figsize=(8, 8))
    #fig.clf()
    labels = "test"
    fig = sky_in_box(fig, az_deg=1, zen_deg=20, label=labels)
    #plt.draw()
    plt.legend()
    plt.show()