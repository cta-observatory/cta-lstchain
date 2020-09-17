'''
New version of a Bokeh camera display, by K. Kosack, see
https://github.com/cta-observatory/ctapipe/issues/1247

Copied here for testing, the ctapipe version of it should replace this code
as soon as it is released!
'''

__all__ = [
    '_generate_polygon_vertices',
    '_generate_bokeh_multi_polygon_arrays',
    'CameraDisplay'
]

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

from ctapipe.instrument import CameraGeometry

import numpy as np
import bokeh
from bokeh.io import output_notebook, push_notebook, show

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
        use_notebook=True, autoshow=True):

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

        self.figure = figure(
            title=f"{geom} ({geom.frame.__class__.__name__})",
            match_aspect=True,
            aspect_scale=1,
        )
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
