import os
import glob
import numpy as np
from shapely import Point, Polygon
from argparse import ArgumentParser

# Internal dependencies:
from lib.plotutils import fmd_histogram
from lib.ioutils import ParameterSet, load_points, load_bins, minmax_in_polygon_file
from compute_ab_values import TruncatedGRestimator


def _set_value_if_not_nan(value, additional_condition=True):
    if np.isnan(value):
        return None
    elif additional_condition:
        return value


if __name__ == "__main__":
    
    # Read input arguments:
    parser = ArgumentParser(description="Plot Frequency-Magnitude Distributions (FMD) for a set of pixels in a zoneless model")
    parser.add_argument("configfile", 
                        help="Configuration file")

    parser.add_argument("index",
                        help="Cell indices",
                        nargs='+',
                        type=int)
    
    parser.add_argument("-g", "--showgrid", 
                        help="Show grid", 
                        action="store_true")

    parser.add_argument('-o', '--output-directory',
                        help="Output directory for figures",
                        default=None)

    args = parser.parse_args()
    
    # Load parameters:
    prms = ParameterSet()
    prms.load_settings(args.configfile)
            
    # Check output directory existence:
    if args.output_directory is None:
        args.output_directory = prms.figures_dir
    if not os.path.isdir(args.output_directory):
        os.mkdir(args.output_directory)

    # Load geographical boundaries:
    envelope = load_points(prms.bounds_file)
           
    # Load seismicity rates and truncated G-R parameters:
    inputfile = os.path.join(prms.output_dir, 'gridded_densities.txt')
    estim = TruncatedGRestimator()
    estim.load_densities(inputfile, scaling_factor=1.0)  # NB: Scaling already applied in voronoi2density.py
    estim.load_bins(prms.bins_file)
    minmags = estim.bins['mins']
    maxmags = estim.bins['maxs']
    grt_prms = np.loadtxt(os.path.join(prms.output_dir, 'ab_values.txt'), delimiter=';')
    if estim.cellinfo is None:
        xy = grt_prms[:, :2]  # Longitudes, Latitudes
    else:
        xy = estim.cellinfo[:, :2]
    if prms.fmd_info_file:
        # Load FMD information (Mmin, Mmax, and optionally bin durations):
        estim.load_fmd_info(prms.fmd_info_file)
    if estim.bin_durations is None:
        estim.bin_durations = np.tile(estim.bins['durations'].reshape((1, estim.nbins)),
                                            (estim.ncells, 1))

    # Plot FMDs for all pixels requested:    
    for indx in args.index:
            centroid = Point(xy[indx, :].tolist())
            print(f">> Plot FMD for pixel {indx} ({centroid.x}, {centroid.y})")
            a = _set_value_if_not_nan(grt_prms[indx, 2])
            b = _set_value_if_not_nan(grt_prms[indx, 3])
            da = _set_value_if_not_nan(grt_prms[indx, 4])

            if np.isinf(a):
                a = None
                b = None
                da = None
                print(f'Warning: Infinite a-value detected. Do not plot FMD model.')

            if prms.fmd_info_file:
                mmax = _set_value_if_not_nan(estim.cellinfo[indx, 3], additional_condition=lambda x: x > 0)
            else:
                mmax = None

            # Reconstruct annual seismicity rates for the current cell (normalized to scaling area):
            cell_intensities = np.reshape(estim.densities[indx, 2:], (estim.nbins, ))
            cell_durations = np.reshape(estim.bin_durations[indx, :], (estim.nbins,))

            # Remove unused bins (with durations <= 0):
            ib = (estim.bin_durations[indx, :] > 0)
            if np.all(cell_intensities == 0):
                print(f'Error: Event counts equal to 0 in every magnitude bin. Skipping.')
                continue

            fmd_histogram(minmags[ib],
                          maxmags[ib],
                          cell_intensities[ib] / cell_durations[ib],
                          envelope,
                          centroid,
                          a=a, 
                          b=b,
                          da=da,
                          show_inset=True, 
                          mmax=mmax, 
                          savefig=os.path.join(args.output_directory, f"fmd_pixel_{indx}.png"),
                          showgrid=args.showgrid)
        
            

    