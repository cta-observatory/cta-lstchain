'''
New version of a Bokeh camera display, by K. Kosack, see
https://github.com/cta-observatory/ctapipe/issues/1247

Copied here for testing, the ctapipe version of it should replace this code
as soon as it is released!
'''

__all__ = [
    '_generate_polygon_vertices',
    '_generate_bokeh_multi_polygon_arrays',
    'CameraDisplay',
    'show_camera',
    'get_pixel_location'
]

import copy
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.models import HoverTool
from bokeh.models import ColumnDataSource, CustomJS, Slider, RangeSlider
from bokeh.models.annotations import Title
from bokeh.models.widgets import Tabs, Panel
from ctapipe.instrument import CameraGeometry
from pkg_resources import resource_filename

import numpy as np
import bokeh
from bokeh.io import output_notebook, push_notebook, show

pixel_hardware_info = []

def _generate_polygon_vertices(order=6, x=0, y=0, size=1.0, theta0=0.0):
    """
    returns the vertices of a regular polygon of order `order`,
    centered at `(x,y)`, with size `size` and rotation angle `theta0` in deg.
    """
    angles = (np.linspace(0.0, 360.0, order + 1) + theta0) * np.pi / 180.0
    return size * np.cos(angles) + x, size * np.sin(angles) + y


def _generate_bokeh_multi_polygon_arrays(geom, order=6):
    pixel_polys_x = []
    pixel_polys_y = []
    for pix_x, pix_y, pix_area in zip(
        geom.pix_x.value, geom.pix_y.value, geom.pix_area.value
    ):
        vx, vy = _generate_polygon_vertices(
            order=order,
            x=pix_x,
            y=pix_y,
            size=np.sqrt(pix_area * 2) * 3 ** (-3 / 4),  # only for hexagon
            theta0=geom.pix_rotation.deg + 180 / order,
        )
        pixel_polys_x.append([[vx]])
        pixel_polys_y.append([[vy]])
    return pixel_polys_x, pixel_polys_y


class CameraDisplay:
    """
    CameraDisplay implementation in Bokeh
    """

    def __init__(self, geom: CameraGeometry, zlow = 0., zhigh = 1.,
                 label='Camera display', title='', use_notebook=True,
                 autoshow=True):

        self._geom = geom
        self._use_notebook = use_notebook
        self._handle = None

        order = 6 if geom.pix_type == "hexagonal" else 4
        xs, ys = _generate_bokeh_multi_polygon_arrays(geom, order=order)

        self.datasource = bokeh.plotting.ColumnDataSource(
            data=dict(
                poly_xs=xs,
                poly_ys=ys,
                image=np.ones_like(geom.pix_x.value).astype(np.float)
            )
        )

        self._color_mapper = bokeh.models.mappers.LinearColorMapper(
                palette=bokeh.palettes.Viridis256, low=zlow, high=zhigh,
                nan_color='white'
        )

        self.figure = figure(title=title, match_aspect=True, aspect_scale=1)
        self.label = label

        # nmodifies the BoxZoomTool. May be better way to get it than by index
        self.figure.toolbar.tools[2].match_aspect = True
        self._setup_camera(geom)

        if use_notebook:
            output_notebook()

        if autoshow:
            self._handle = show(self.figure, notebook_handle=use_notebook)

    def _setup_camera(self, geom):
        self._pixels = self.figure.multi_polygons(
            xs="poly_xs",
            ys="poly_ys",
            fill_color=bokeh.transform.transform("image", self._color_mapper),
            line_color=None,
            source=self.datasource,
        )
        self._color_bar = bokeh.models.ColorBar(
            color_mapper=self._color_mapper,
            label_standoff=12,
            border_line_color=None,
            location=(0, 0),
        )
        self.figure.add_layout(self._color_bar, "right")
        self.figure.add_layout(Title(text=self.label, align='left'), 'above')

    def update(self):
        if self._use_notebook and self._handle:
            push_notebook(self._handle)

    def rescale(self, percent=100):
        self._color_mapper.update(
            low=self.datasource.data["image"].min(),
            high=(percent / 100) * self.datasource.data["image"].max(),
        )

    def set_geom_and_image(self, geom, image):
        self.geom = geom
        self.image = image

    @property
    def geom(self):
        return self._geom

    @geom.setter
    def geom(self, new_geom):
        self._geom = new_geom

    @property
    def image(self,):
        return self.datasource.data["image"]

    @image.setter
    def image(self, new_image):
        self.datasource.data["image"] = new_image
        self.update()

def show_camera(content, geom, pad_width, pad_height, label, titles=None):

    """

    Parameters
    ----------
    content: pixel-wise quantity to be plotted, ndarray with shape (N,
    number_of_pixels) where N is the number of different sets of pixel
    values, for example N different data runs or whatever. The shape can also
    be just (number_of_pixels), in case a single camera display is to be shown

    geom: camera geometry
    pad_width: width in pixels of each of the 3 pads in the plot
    pad_height: height in pixels of each of the 3 pads in the plot
    label: string to label the quantity which is displayed, the same for the N
    sets of pixels inside "content"
    titles: list of N strings, with the title specific to each of the sets
    of pixel values to be displayed: for example, indicating run numbers

    Returns
    -------
    [p1, p2, p3]: three bokeh figures, intended for showing them on the same row
    p1 is the camera display (with "content" in linear & logarithmic scale)
    p2: content vs. pixel
    p3: histogram of content (with one entry per pixel)

    """

    # patch to reduce gaps between bokeh's cam circular pixels:
    camgeom = copy.deepcopy(geom)
    #camgeom.pix_area *= 1.3

    numsets = 1
    if np.ndim(content) > 1:
        numsets = content.shape[0]
    # numsets is the number of different sets of pixel data to be displayed

    allimages = []
    if np.ndim(content) == 1:
        allimages.append(content)
    else:
        for i in range(1,numsets+1):
            allimages.append(content[i-1])

    if titles is None:
        titles = [None]*numsets

    cam = CameraDisplay(camgeom, np.nanmin(allimages), np.nanmax(allimages),
                        label, titles[0], use_notebook=False, autoshow=False)
    cam.image = allimages[0]
    cam.figure.title.text = titles[0]

    allimageslog = []

    for image in allimages:
        logcontent = np.copy(image)
        for i, x in enumerate(logcontent):
            # workaround as long as log z-scale is not implemented in bokeh camera:
            if x <= 0:
                logcontent[i] = np.nan
            else:
                logcontent[i] = np.log10(image[i])
        allimageslog.append(logcontent)

    camlog = CameraDisplay(camgeom, np.nanmin(allimageslog),
                           np.nanmax(allimageslog), label, titles[0],
                           use_notebook=False, autoshow=False)
    camlog.image = allimageslog[0]
    camlog.figure.title.text = titles[0]

    cluster_i = []
    cluster_j = []
    pix_id_in_cluster = []
    for i in camgeom.pix_id:
        data = get_pixel_location(i)
        cluster_i.append(data[0])
        cluster_j.append(data[1])
        pix_id_in_cluster.append(data[2])

    for c in [cam, camlog]:
        c.datasource.add(list(c.geom.pix_id), 'pix_id')
        c.datasource.add(cluster_i, 'cluster_i')
        c.datasource.add(cluster_j, 'cluster_j')
        c.datasource.add(pix_id_in_cluster, 'pix_id_in_cluster')

        # c.add_colorbar()
        c.figure.plot_width = pad_width
        c.figure.plot_height = int(pad_height * 0.85)
        c.figure.grid.visible = False
        c.figure.axis.visible = True
        c.figure.xaxis.axis_label = 'X position (m)'
        c.figure.yaxis.axis_label = 'Y position (m)'
        c.figure.add_tools(
            HoverTool(tooltips=[('pix_id', '@pix_id'),
                                ('value', '@image'),
                                ('cluster (i,j)', '(@cluster_i, @cluster_j)'),
                                ('pix # in cluster', '@pix_id_in_cluster')],
                      mode='mouse', point_policy='snap_to_data'))


    tab1 = Panel(child=cam.figure, title='linear')
    tab2 = Panel(child=camlog.figure, title='logarithmic')

    p1 = Tabs(tabs=[tab1, tab2])
    p1.margin = (0, 0, 0, 25)

    minmax = np.nanmax(allimages) - np.nanmin(allimages)
    p2 = figure(background_fill_color='#ffffff',
                y_range=(np.nanmin(allimages)-0.1*minmax,
                         np.nanmax(allimages)+0.1*minmax),
                x_axis_label='Pixel id',
                y_axis_label=label)
    p2.min_border_top = 60
    p2.min_border_bottom = 70

    source2 = ColumnDataSource(data=dict(pix_id=cam.geom.pix_id,
                                         value=cam.image))
    p2.circle(x='pix_id', y='value', size=2, source=source2)
    p2.add_tools(
        HoverTool(tooltips=[('(pix_id, value)', '(@pix_id, @value)')],
                  mode='mouse', point_policy='snap_to_data'))

    allhists = []
    alledges = []
    for image in allimages:
        hist, edges = np.histogram(image[~np.isnan(image)], bins=200)
        allhists.append(hist)
        alledges.append(edges)

    source3 = ColumnDataSource(data=dict(top=allhists[0],
                                         bottom=0.7*np.ones_like(allhists[0]),
                                         left=alledges[0][:-1],
                                         right=alledges[0][1:]))

    p3 = figure(background_fill_color='#ffffff',
                y_range=(0.7, np.max(allhists) * 1.1),
                x_range=(np.nanmin(allimages)-0.1*minmax,
                         np.nanmax(allimages)+0.1*minmax),
                x_axis_label=label,
                y_axis_label='Number of pixels', y_axis_type='log')
    p3.quad(top='top', bottom='bottom', left='left', right='right',
            source=source3)

    if titles is None:
        titles = [None]*len(allimages)

    cds_allimages = ColumnDataSource(data=dict(z=allimages, zlog=allimageslog,
                                               hist=allhists, edges=alledges,
                                               titles=titles))

    callback = CustomJS(args=dict(source1=cam.datasource,
                                  source1log = camlog.datasource,
                                  source2=source2, source3=source3,
                                  zz=cds_allimages,
                                  title=cam.figure.title,
                                  titlelog=camlog.figure.title),
    code="""
        var slider_value = cb_obj.value
        var z = zz.data['z']
        var zlog = zz.data['zlog']
        var edges = zz.data['edges']
        var hist = zz.data['hist']
        for (var i = 0; i < source1.data['image'].length; i++) {
             source1.data['image'][i] = z[slider_value-1][i]
             source1log.data['image'][i] = zlog[slider_value-1][i]
             source2.data['value'][i] = source1.data['image'][i]
        }
        for (var i = 0; i < source3.data['top'].length; i++) {
             source3.data['top'][i] = hist[slider_value-1][i]
             source3.data['left'][i] = edges[slider_value-1][i]
             source3.data['right'][i] = edges[slider_value-1][i+1]
        }
        title.text = zz.data['titles'][slider_value-1]
        titlelog.text = title.text
        source1.change.emit();
        source1log.change.emit();
        source2.change.emit();
        source3.change.emit();
    """)

    slider = None
    if numsets > 1:
        slider = Slider(start=1, end=numsets, value=1, step=1, title="run",
                        orientation='vertical', show_value=False, height=300)
        slider.margin = (0, 0, 0, 35)
        slider.js_on_change('value', callback)

    callback2 = CustomJS(args=dict(color_mapper=cam._color_mapper,
                                   color_mapper_log=camlog._color_mapper),
    code="""
        var range = cb_obj.value
        color_mapper.low = range[0]
        color_mapper.high = range[1]
        if (range[0] > 0.)
            color_mapper_log.low = Math.log(range[0])/Math.LN10    
        color_mapper_log.high = Math.log(range[1])/Math.LN10
        color_mapper.change.emit();
        color_mapper_log.change.emit();
    """)
    step = (np.nanmax(allimages) - np.nanmin(allimages))/100.
    range_slider = RangeSlider(start=np.nanmin(allimages),
                               end=np.nanmax(allimages),
                               value=(np.nanmin(allimages),
                                      np.nanmax(allimages)), step=step,
                               title="z_range", orientation='vertical',
                               direction='rtl', height=300,
                               show_value=False)
    range_slider.js_on_change('value', callback2)

    return [slider, p1, range_slider, p2, p3]


def get_pixel_location(pix_id):

    """

    Parameters
    ----------
    pix_id pixel id number

    Returns
    -------
    "Hardware" parameters of the pixel:
    [cluster_i, cluster_j, pixe_id_within_cluster]

    Info from https://forge.in2p3.fr/issues/33587, stored in
    cta-lstchain/lstchain/io/LST_pixid_to_cluster.txt

    """
    if len(pixel_hardware_info) > 0:
        return pixel_hardware_info[pix_id]

    # The first time we read in the data stored in the resources directory:
    infilename = resource_filename('lstchain',
                                   'resources/LST_pixid_to_cluster.txt')
    data = np.genfromtxt(infilename, comments='#',dtype='int')

    pixel_hardware_info.extend([None]*(1 + data[:,0].max()))
    for d in data:
        pixel_hardware_info[d[0]] = [d[1], d[2], d[3]]

    return pixel_hardware_info[pix_id]
