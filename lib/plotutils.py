"""
PLOTTING ROUTINES FOR ZONELESS_VORONOI.PY
"""
import sys
from os import remove
from math import ceil, floor
import pygmt
from shapely import Point, Polygon, MultiPolygon, MultiPoint
import numpy as np
from tqdm import tqdm

from lib.grt_ml import cumulative_fmd



class ProgressBar():
    """
    Progress-bar object definition
    """
    def __init__(self, imax: float, title: str='', nsym: int=20):
        """
        :param imax: float, Maximum counter value, corresponding to 100% advancment
        :param title: str, (Optional) Title string for the progress bar
        :param nsym: int, (Optional) Width of progress bar, in number of "=" symbols (default: 20)
        """
        self.imax = imax
        self.title = title
        self.nsym = nsym


    def update(self, i: float, imax: float=None, title: str=None):
        """ Display an ASCII progress bar with advancement level at (i/imax) %

        :param i: float, Current counter value
        :param imax: float, Maximum counter value, corresponding to 100% advancment
        :param title: str, (Optional) Title string for the progress bar
        :param nsym: int, (Optional) Width of progress bar, in number of "=" symbols (default: 20)
        """
        if imax is not None:
            self.imax = imax
        if title is not None:
            self.title = title
        sys.stdout.write('\r')
        fv = float(i)/float(self.imax)  # Fractional value, between 0 and 1
        sys.stdout.write( ('{0} [{1:'+str(self.nsym)+'s}] {2:3d}%').format(self.title, '='*ceil(
            fv*self.nsym), floor(fv*100)) )
        if i==self.imax:
            sys.stdout.write('\n')
        sys.stdout.flush()


def map_polygons(polygons_file: str,
                 clim=None,
                 bounds = None,
                 colormap='red2green', 
                 colormap_reversal=False, 
                 colormap_nbins=30, 
                 cbar_title = f"Earthquake density",
                 savefig=False, 
                 filename='cells.png', 
                 showfig=True, 
                 coast_resolution="i", 
                 title=None, 
                 dpi=300,
                 spatial_buffer=1.0,
                 pen="0.5p",
                 add_polygon: Polygon=None,
                 map_projection: str = "M15c",
                 logscale=False,
                 figframe="a",
                 transparency_index=50,
                 add_points=None):
    """
    Draw a geographical map of Voronoi cells, colored as a function of field -Z in input file.
    If clim=None, coloring of polygons is deactivated.

    polygons_file: str, path to ASCII file containing polygons geometries saved in GMT format
    """
    if isinstance(bounds, Polygon):
        xb, yb = bounds.exterior.xy
        limits = [min(xb), max(xb), min(yb), max(yb)]
    elif isinstance(bounds, list) or isinstance(bounds, np.ndarray):
        # Format: [min(x), max(x), min(y), max(y)]
        limits = bounds
    elif bounds is None:
        limits = pygmt.info(polygons_file, per_column=True)
        if isinstance(add_polygon, Polygon):
            # Update bounding box when add_polyghon is specified:
            xe, ye = add_polygon.exterior.xy
            limits = [min(limits[0], min(xe)),
                      max(limits[1], max(xe)),
                      min(limits[2], min(ye)),
                      max(limits[3], max(ye))]

    if title is not None:  # Add title
        figframe = [figframe, f'+t"{title}"']

    if clim:
        if logscale:
            zlim_str = f'{min(clim)}/{max(clim)}/3+l'
        else:
            zlim_str = f'{min(clim)}/{max(clim)}/{colormap_nbins}+n'
        pygmt.makecpt(cmap=colormap, 
                      reverse=colormap_reversal, 
                      series=zlim_str,
                      background="o")

    fig = pygmt.Figure()
    mapbounds = [limits[0] - spatial_buffer, 
                 limits[1] + spatial_buffer,
                 limits[2] - spatial_buffer, 
                 limits[3] + spatial_buffer]
    fig.basemap(projection=map_projection, 
                region=mapbounds, 
                frame=figframe)
    if isinstance(coast_resolution, str):
        fig.coast(borders=["1/0.5p,black"],
                  shorelines=["1/0.5p,black"],
                  resolution=coast_resolution,
                  rivers=["1/0.5p,slategray1"],
                  lakes=["slategray1"])
    if add_polygon:
        if not isinstance(add_polygon, Polygon):
            raise ValueError('Input argument "add_polygon" must be Shapely.Polygon object')
        xp, yp = add_polygon.exterior.coords.xy
        fig.plot(x=xp,
                 y=yp,
                 straight_line=True,
                 pen="1p,black",
                 close=True)
    if clim:
        fig.plot(data=polygons_file, 
                 straight_line=True,
                 zvalue=True, 
                 pen=pen, 
                 close=True, 
                 fill='+z', 
                 cmap=True,
                 transparency=transparency_index)
    else:
        fig.plot(data=polygons_file, 
                 straight_line=True,
                 pen=pen, 
                 close=True, 
                 fill=None)

    if add_points:
        x = [g.x for g in add_points.geoms]
        y = [g.y for g in add_points.geoms]
        fig.plot(x=x,
                 y=y,
                 style='c2p',
                 fill='black')
    if clim:
        fig.colorbar(frame=f"x+l{cbar_title}")
    if savefig:
        if isinstance(savefig, str):
            filename = savefig
        fig.savefig(filename, dpi=dpi)
        print(f'Figure saved in {filename}')
    if showfig:
        fig.show()
    

def fmd_histogram(minmags, maxmags, rates, bounding_polygon: Polygon, centroid_or_polygon: Point,
             a=None, b=None, da=None, show_inset=False, mmax=None, savefig=True, showgrid=True):
    """
    Plot the magnitude-frequency distribution for a single pixel of the zoneless model.

    :param minmags: np.ndarray, minimum bounds of each magnitude bin
    :param maxmags: np.ndarray, maximum bounds of each magnitude bin
    :param rates: np.ndarray, earthquake rate in each magnitude bin (non-cumulative)
    :param bounding_polygon: shapely.Polygon, bounding polygon
    :param centroid_or_polygon: shapely.Point or shapely.Polygon, Cell centroid or area
    :param a, b: float, optional, provide Gutenberg-Richter parameters to superimpose a model on observations.
        Specify a and b values, implies to set "normalized_rates" as True.
    :param da: float, uncertainty on parameter a
    :param show_inset: bool, specify whether to plot current polygon location in inset, or not.
    :param mmax: Optional, None or float, specify whether FMD should be truncated (if float), and
        at which value to truncate. 
    :param savefig: optional, boolean or str, Specify whether to save plot as figure (if True or str). If str 
        then use it as output figure name (Default: "mfd.png")
    
    :returns None
    """
    xb, yb = bounding_polygon.exterior.xy
    limits = [min(xb), max(xb), min(yb), max(yb)]
        
    if (a is None) or (b is None):
        draw_model = False
    else:
        draw_model = True

    if mmax is None:
        mmax = np.inf
    incremental = np.squeeze(rates)
    cumulative = np.flipud(np.cumsum(np.flipud(incremental)))
    igt0 = cumulative > 0.0
    imax = np.where(igt0)[0][np.argmin(cumulative[igt0])]

    if draw_model:
        if da:
            print(f'a = {a} +/- {da}  b = {b}')
        else:
            print(f'a = {a}   b = {b}')
        dm = maxmags[0] - minmags[0]
        # Compute theoretical FMD at central values :
        model = cumulative_fmd(minmags + dm / 2, a, b, dm=dm, mmax=mmax)
        if da:
            model_lower = cumulative_fmd(minmags + dm / 2, a - da, b, dm=dm, mmax=mmax)
            model_upper = cumulative_fmd(minmags + dm / 2, a + da, b, dm=dm, mmax=mmax)

    bounds = [np.floor(minmags.min()) - 1,
              np.ceil(maxmags.max()) + 1,
              np.power(10, np.log10(cumulative[:imax].min()) - 1),
              np.power(10, np.log10(cumulative[:imax].max()) + 1)]

    if showgrid:
        plot_frame = ['xag+lMagnitude', 
                      'ya1f3g3+l"Annual rate of exceedance"',
                      f'WSne']
    else:
        plot_frame = ['xa+lMagnitude', 
                      'ya1f3+l"Annual rate of exceedance"',
                      f'WSne']

    filename = "mfd.png"
    if isinstance(savefig, str):
        filename = savefig
        savefig = True

    fig = pygmt.Figure()
    fig.basemap(projection="X10c/10cl",   # Linear-Log projection
                region=bounds, 
                frame=plot_frame)
    fig.plot(x=minmags, y=incremental, style=f'c6p', pen='red', label='incremental')  # Incremental FMD
    fig.plot(x=minmags, y=cumulative, style=f'c8p', fill='red', label='cumulative')  # Cumulative FMD
    if draw_model:
        if da:
            fig.plot(x=minmags, y=model_lower, pen="thinner,black,.")
            fig.plot(x=minmags, y=model_upper, pen="thinner,black,.")
        #fig.plot(x=minmags0, y=model, pen="thick,black,-", label=f'a={a:.2f} b={b:.2f}')
        fig.plot(x=minmags, y=model, pen="thick,black,solid", label=f'a={a:.2f} b={b:.2f}')
        fig.legend()

    if show_inset:
        # Display the polygon position (in red) over the whole model area in the inset:
        with fig.inset(position="jBL+o0.3c/0.3c",
                        box="+pblack",
                        region=limits,
                        projection='M3c'):
            fig.coast(
                land="gray",
                borders=1,
                resolution="i",
                water="white")
 
            if isinstance(centroid_or_polygon, Point):
                fig.plot(x=centroid_or_polygon.x,
                         y=centroid_or_polygon.y,
                         style="s0.1c",
                         pen="0.2p,black,solid",
                         fill="red")
            elif isinstance(centroid_or_polygon, Polygon):
                xx, yy = centroid_or_polygon.exterior.coords.xy
                fig.plot(x=xx,
                         y=yy,
                         close=True,
                         pen="0.2p,black,solid",
                         fill="red")

    if savefig:
        fig.savefig(filename)
        print(f'Figure saved in {filename}')
    else:
        fig.show()


def empirical_distribution(values, nbins, bounding_polygon: Polygon, polygon: Polygon=None,
                           draw_line=True, show_markers=False, show_inset=False, savefig=True,
                           showgrid=True, showfig=False, append_to_fig=None, clim=None):
    """
    Plot the magnitude-frequency distribution for a single pixel of the zoneless model.

    :param values: np.ndarray, sample values
    :param nbins: int, number of bins for the distribution of values
    :param bounding_polygon: shapely.Polygon, bounding polygon
    :param polygon: shapely.Polygon, [optional] Contour of the current polygon. Default: None.
    :param draw_line: bool, [optional] plot distribution as a line . Default: True.
    :param show_markers: bool, [optional] plot distribution as markers . Default: False.
    :param show_inset: bool, [optional] plot current polygon location in inset. Default: False.
    :param savefig: boolean or str, [optional] If True, save figure into file. If str, use this
        string as output filename, otherwise use the default name "distrib.png".
    :param showgrid: bool, [optional] Add grid on plot. Default: True.
    :param showfig: bool, [optional] Opens figure.
    :param append_to_fig: pygmt.Figure, [optional] Specify whether to superimpose distribution on
        an existing pygmt.Figure instance. Default: None
    :param clim: list or np.ndarray, [optional] X-axis bounds. Default: None, use [values.min(), values.max()]

    :returns pygmt.Figure() instance
    """
    xb, yb = bounding_polygon.exterior.xy
    limits = [min(xb), max(xb), min(yb), max(yb)]

    pdf, edges = np.histogram(values, bins=nbins, density=True)
    mean_value = np.nanmean(values)
    x = edges[:-1] + 0.5 * np.diff(edges)

    if clim is not None:
        bounds = [min(clim), max(clim),
                  0, pdf.max() * 1.1]
    else:
        xmin = np.nanmin(values)
        xmax = np.nanmax(values)
        bounds = [max(0, xmin - 0.15 * (xmax - xmin)), xmax + 0.25 * (xmax - xmin),
                  0, pdf.max() * 1.1]

    if showgrid:
        plot_frame = ['xag+lValues',
                      'ya1f3g3+l"PDF"',
                      f'WSne']
    else:
        plot_frame = ['xa+lValues',
                      'ya1f3+l"PDF"',
                      f'WSne']

    filename = "distrib.png"
    if isinstance(savefig, str):
        filename = savefig
        savefig = True

    if append_to_fig is None:
        fig = pygmt.Figure()
        fig.basemap(projection="X10c",  # Linear-Linear projection
                    region=bounds,
                    frame=plot_frame)
    else:
        fig = append_to_fig

    if draw_line:
        fig.plot(x=x, y=pdf, pen='1p,black,solid')
    if show_markers:
        fig.plot(x=x, y=pdf, style=f'c6p', fill='black')
    fig.plot(x=[mean_value, mean_value], y=bounds[2:], pen='0.5p,black,dotted', label=f'mean = {mean_value:.1f}')
    fig.legend()

    if show_inset and (polygon is not None):
        # Display the polygon position (in red) over the whole model area in the inset:
        with fig.inset(position="jBL+o8.2c/0.3c",
                       box="+pblack",
                       region=limits,
                       projection='M1.5c'):
            fig.coast(
                land="gray",
                borders=1,
                resolution="i",
                water="white")
            xp, yp = polygon.exterior.xy
            fig.plot(x=xp,
                     y=yp,
                     style="s0.1c",
                     pen="0.2p,black,solid",
                     fill="red")

    if savefig:
        fig.savefig(filename)
        print(f'Figure saved in {filename}')
    elif showfig:
        fig.show()
    else:
        return fig


def map_bounds_and_cells(bounds: Polygon, cells: Polygon, events: MultiPoint = None,
                         filename='bounds_and_cells.png'):
    x1b, y1b, x2b, y2b = bounds.bounds
    x1c, y1c, x2c, y2c = cells.bounds
    fig = pygmt.Figure()
    fig.basemap(projection='M10c',
                region=[min(x1b, x1c), max(x2b, x2c), min(y1b, y1c), max(y2b, y2c)],
                frame='a')

    if events is not None:
        # Show events:
        has_label = False
        xe = [e.x for e in events.geoms]
        ye = [e.y for e in events.geoms]
        if not has_label:
            labelstr = 'events'
            has_label = True
        else:
            labelstr = None
        fig.plot(x=xe, y=ye, style='p2p', fill='gray', label=labelstr)

    has_label = False
    for c in tqdm(cells.geoms):
        xc, yc = c.exterior.xy
        if not has_label:
            labelstr = f"cells (area: {cells.area})"
            has_label = True
        else:
            labelstr = None
        fig.plot(x=xc, y=yc, pen="1p,black,solid", label=labelstr, close=True)

    xb, yb = bounds.exterior.xy
    fig.plot(x=xc, y=yc, pen="1p,red,dashed", label=f"bounds (area: {bounds.area})", close=True)

    fig.legend()
    fig.savefig(filename)
    print(f'Figure of bounds and cells saved in {filename}')
