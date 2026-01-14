import os
import glob
import numpy as np
from scipy import stats
import openturns as ot
from shapely import Point, Polygon
from argparse import ArgumentParser
from tqdm import tqdm

# Internal dependencies:
from lib.plotutils import empirical_distribution
from lib.ioutils import (ParameterSet,
                         load_points,
                         load_bins,
                         minmax_in_polygon_file,
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
    
    parser.add_argument("-g", "--showgrid", 
                        help="Show grid", 
                        action="store_true")

    parser.add_argument('-o', '--output-directory',
                        help="Output directory for figures",
                        default=None)

    parser.add_argument('-n', '--number-of-bins',
                        help="Number of bins used per axis to plot the distribution",
                        type=int,
                        default=20)

    parser.add_argument("-r", "--range",
                        help="Specify lower and upper limits for the X- and Y-axes. 4 arguments in the Xmin, Xmax, Ymin, Ymax order",
                        nargs=4,
                        type=float,
                        default=None)

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

    # Load geographical boundaries:
    envelope = load_points(prms.bounds_file)
    xb, yb = envelope.exterior.xy
    limits = [min(xb), max(xb), min(yb), max(yb)]

    # Load current polygon, or pixel:
    multipoly, _ = load_polygons(args.polygons_file)
    polygon = multipoly.geoms[args.index - 1]

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
        a, b, da, db, cov, lonlat = load_ab_from_grid(file, cell_index, return_centroid=True)  # 1-based cell-index
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
        distribs.append(ot.Normal([a, b], cov))
    mixture = ot.Mixture(distribs)  # Mixture distribution with equal weights for every element

    # If needed, define range from extreme sample values:
    if args.range is None:
        arange = [amin - 3 * damax, amax + 3 * damax]
        brange = [bmin - 3 * dbmax, bmax + 3 * dbmax]
    else:
        arange = args.range[0:2]
        brange = args.range[2:4]

    # Sample distribution over the 2-D plane:
    bg, ag = np.meshgrid(np.linspace(brange[0], brange[1], args.number_of_bins),
                         np.linspace(arange[0], arange[1], args.number_of_bins),
                         indexing='xy')
    bf = bg.flatten()
    af = ag.flatten()
    zf = np.array([mixture.computePDF(ot.Point([b, a])) for a, b in zip(bf, af)])
    zmin = zf.min()
    zmax = zf.max()
    zrange = zmax - zmin

    # Make plot:
    print(f">> Plot distribution for cell {args.index}, with centroid ({lonlat[0]}, {lonlat[1]})")
    grd = pygmt.xyz2grd(x=xf,
                         y=yf,
                         z=zf,
                         region=brange + arange,
                         spacing=((brange[1] - brange[0]) / args.number_of_bins, (arange[1] - arange[0]) / args.number_of_bins),
                         outgrid='grid.nc')
    pygmt.makecpt(cmap='roma', reverse=True, series=f'{zmin}/{zmax}/{0.01 * zrange}', background=True)

    # --> 2-D probability distribution function:
    fig = pygmt.Figure()
    fig.basemap(projection="X15c", frame=["a"], region=brange + arange)
    fig.grdimage(grid=grd, cmap=True)
    if args.draw_contours:
        grd_interval = zrange / 10
        grd_annotation = "-"  # Disable all annotations
        fig.grdcontour(grid=grd, levels=grd_interval, annotation=grd_annotation)
    fig.colorbar(cmap=True, frame=f"xa+l{PDF}")

    # --> Add inset with the polygon location (in red) over the whole model area in the inset:
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

    fig.savefig(os.path.join(args.output_directory, f"ab_aggregate_distrib_cell_{args.index}.png"), dpi=300)
