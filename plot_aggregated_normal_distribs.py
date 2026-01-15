import os
import glob
import numpy as np
from scipy import stats
import openturns as ot
from shapely import Point, Polygon
from argparse import ArgumentParser
from tqdm import tqdm
import pygmt

# Internal dependencies:
from lib.ioutils import (ParameterSet,
                         load_points,
                         load_bins,
                         load_polygons)


def load_ab_from_grid(filename, cell_index, return_centroid=False):
    """
    Returns a, b, da, db, cov values from ab_values.txt formatted-like file.

    :param filename: str, Path to ab_values.txt file
    :param cell_index: int, 1-based cell index
    """
    grid = np.loadtxt(filename, delimiter=';')
    a, b, da, db, rho = grid[cell_index - 1, 2:7]
    lonlat = grid[cell_index - 1, 0:2]
    cov = ot.CovarianceMatrix(2, [da ** 2, rho * da * db, rho * da * db, db ** 2])
    if return_centroid:
        return a, b, da, db, cov, lonlat
    else:
        return a, b, da, db, cov



if __name__ == "__main__":
    
    # Read input arguments:
    parser = ArgumentParser(description="Plot aggregated normal distributions from a- and b-values in a collection of files formatted like  'ab_values.txt'")
    parser.add_argument("configfile", 
                        help="Configuration file")

    parser.add_argument("-i", "--index",
                        help="Cell index (required, single value)",
                        required=True,
                        type=int)

    parser.add_argument("-p", "--polygons-file",
                        help="Path to GMT-formatted file with polygon cells geometries, e.g., a_cells.txt",
                        required=True,
                        type=str)

    parser.add_argument("files",
                        help="input files",
                        nargs='+',
                        type=str)

    parser.add_argument('-o', '--output-directory',
                        help="Output directory for figures",
                        default=None)

    parser.add_argument('-n', '--number-of-bins',
                        help="Number of bins used per axis to plot the distribution",
                        type=int,
                        default=100)

    parser.add_argument("-r", "--range",
                        help="Specify lower and upper limits for the X- and Y-axes. 4 arguments in the Xmin, Xmax, Ymin, Ymax order",
                        nargs=4,
                        type=float,
                        default=None)

    parser.add_argument("-c", "--colormap",
                        help="Specify GMT colormap explicitely. Note that colormap are increasing in reverse order here.",
                        type=str,
                        default="roma")

    parser.add_argument("--draw-contours",
                        help="Overlay contours on 2-D PDF plot",
                        action='store_true')

    args = parser.parse_args()
    
    # Load parameters:
    prms = ParameterSet()
    prms.load_settings(args.configfile)

    # Check output directory existence:
    if args.output_directory is None:
        args.output_directory = prms.figures_dir
    if not os.path.isdir(args.output_directory):
        print(f'>> Creating output directory: "{args.output_directory}"')
        os.mkdir(args.output_directory)

    # Load current polygon, or pixel:
    multipoly, _ = load_polygons(args.polygons_file)
    polygon = multipoly.geoms[args.index - 1]

    # Load geographical boundaries:
    if prms.bounds_file is None:
        limits = [multipoly.bounds[0], multipoly.bounds[2], multipoly.bounds[1], multipoly.bounds[3]]
    else:
        envelope = load_points(prms.bounds_file)
        xb, yb = envelope.exterior.xy
        limits = [min(xb), max(xb), min(yb), max(yb)]

    # Load values and build mixture distribution:
    values = []
    cnt = 0
    if len(args.files) > 1:
        filelist = args.files
    elif len(args.files) == 1:
        filelist = glob.glob(args.files[0])
    n = len(filelist)
    print(f'>> Number of input files: {n}')

    distribs = []
    amin = bmin = np.inf
    amax = bmax =-np.inf
    damin = dbmin = np.inf
    damax = dbmax = -np.inf
    for file in tqdm(filelist):
        a, b, da, db, cov, lonlat = load_ab_from_grid(file, args.index, return_centroid=True)  # 1-based cell-index
        if a > amax:
            amax = a
        if b > bmax:
            bmax = b
        if a < amin:
            amin = a
        if b < bmin:
            bmin = b
        if da > damax:
            damax = da
        if db > dbmax:
            dbmax = db
        try:
            distribs.append(ot.Normal([a, b], cov))
        except TypeError:
            pass
    mixture = ot.Mixture(distribs)  # Mixture distribution with equal weights for every element

    # If needed, define range from extreme sample values:
    if args.range is None:
        arange = [amin - damax, amax + damax]
        brange = [bmin - dbmax, bmax + dbmax]
    else:
        brange = args.range[0:2]
        arange = args.range[2:4]

    # Print mxiture distribution statistics:
    mean  = mixture.getMean()
    std = mixture.getStandardDeviation()
    corrcoef = mixture.getPearsonCorrelation()[0,1]
    print(f'>> Mixture distribution:\n\tmeans = {mean}\n\tstd.devs. = {std}\n\trho = {corrcoef}')

    # Sample distribution over the 2-D plane:
    bg, ag = np.meshgrid(np.linspace(brange[0], brange[1], args.number_of_bins + 1),
                         np.linspace(arange[0], arange[1], args.number_of_bins + 1),
                         indexing='xy')
    bf = bg.flatten()
    af = ag.flatten()
    zf = np.array([mixture.computePDF(ot.Point([b, a])) for a, b in zip(bf, af)])
    zmin = zf.min()
    zmax = zf.max()
    zrange = zmax - zmin

    # Make plot:
    print(f">> Plot distribution for cell {args.index}, with centroid ({lonlat[0]}, {lonlat[1]})")
    dx = (brange[1] - brange[0]) / args.number_of_bins
    dy = (arange[1] - arange[0]) / args.number_of_bins
    grd = pygmt.xyz2grd(x=bf,
                        y=af,
                        z=zf,
                        region=[brange[0] - dx, brange[1] + dx, arange[0] - dy, arange[1] + dy],
                        spacing=f'{dx}/{dy}')
    pygmt.makecpt(cmap=args.colormap, reverse=True, series=f'{zmin}/{zmax}/{0.01 * zrange}', background=True)

    # --> 2-D probability distribution function:
    fig = pygmt.Figure()
    fig.basemap(projection="X15c", frame=["a"], region=[brange[0], brange[1], arange[0], arange[1]])
    fig.grdimage(grid=grd, cmap=True)
    if args.draw_contours:
        grd_interval = zrange / 10
        grd_annotation = "-"  # Disable all annotations
        fig.grdcontour(grid=grd, levels=grd_interval, annotation=grd_annotation)
    fig.plot(x=[mean[1], mean[1]], y=[arange[0], mean[0]], pen='1p,black,dotted')
    fig.plot(x=[brange[0], mean[1]], y=[mean[0], mean[0]], pen='1p,black,dotted', label='Mean')
    fig.legend()
    fig.colorbar(cmap=True, frame="xa+lPDF")

    # --> Add inset with the polygon location (in red) over the whole model area in the inset:
    with fig.inset(position="jBL+o12.0c/0.3c",
                   box="+pblack",
                   region=limits,
                   projection='M3.0c'):
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

    fig.savefig(os.path.join(args.output_directory, f"ab_aggregate_distrib_cell_{args.index}.png"), dpi=300)
