import os
import glob
import numpy as np
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


if __name__ == "__main__":
    
    # Read input arguments:
    parser = ArgumentParser(description="Plot distribution(s) of Z-values in GMT multiple segment files")
    parser.add_argument("configfile", 
                        help="Configuration file")

    parser.add_argument("-i", "--index",
                        help="Cell index (required, O-based indexing)",
                        required=True,
                        type=int)

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

    parser.add_argument('-s', '--single-plot',
                        help="Plot distributions for all indices on the same graph",
                        action="store_true")

    parser.add_argument('-n', '--number-of-bins',
                        help="Number of bins used to compute the distribution",
                        type=int,
                        default=20)

    parser.add_argument("-r", "--range",
                        help="Specify lower and upper limits for the X-axis (in linear scale if -l is activated)",
                        nargs=2,
                        type=float,
                        default=None)

    parser.add_argument("-l", "--log-input-data",
                        help="Transform input Z-values from a logarithmic scale (base 10) to a linear scale",
                        action="store_true")

    args = parser.parse_args()
    
    # Load parameters:
    prms = ParameterSet()
    prms.load_settings(args.configfile)
    bsdir = os.path.join(prms.output_dir, 'bootstrap', '')
    if not os.path.isdir(bsdir):
        raise ValueError(f'Unknown directory: {bsdir}')

    # Check output directory existence:
    if args.output_directory is None:
        args.output_directory = prms.figures_dir
    if not os.path.isdir(args.output_directory):
        print(f'>> Creating output directory: "{args.output_directory}"')
        os.mkdir(args.output_directory)

    # Load geographical boundaries:
    envelope = load_points(prms.bounds_file)
           
    # Load values:
    values = []
    cnt = 0
    if len(args.files) > 1:
        filelist = args.files
    elif len(args.files) == 1:
        filelist = glob.glob(args.files[0])
    print(f'>> Number of input files: {len(filelist)}')
    for file in tqdm(filelist):
        multipoly, z_values = load_polygons(file)
        if cnt == 0:
            bname = os.path.splitext(os.path.split(file)[1])[0].split('bs_')[0]
            polygons = multipoly.geoms[args.index]
            cnt += 1
        if args.log_input_data:
            values.append(np.power(10, z_values[args.index]))
        else:
            values.append(z_values[args.index])
    arr = np.array(values)

    # Plot distributions for all pixels requested:
    clim = args.range
    current_polygon = multipoly.geoms[args.index]
    centroid = current_polygon.centroid
    print(f">> Plot distribution for cell {args.index}, with centroid ({centroid.x}, {centroid.y})")
    empirical_distribution(arr, args.number_of_bins, envelope, current_polygon,
                           draw_line=True,
                           show_markers=False,
                           show_inset=True,
                           savefig=os.path.join(args.output_directory, f"distrib_pixel_{args.index}.png"),
                           showgrid=args.showgrid,
                           showfig=False,
                           append_to_fig=None,
                           clim=clim)