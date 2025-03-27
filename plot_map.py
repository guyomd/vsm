import os

import numpy as np
from shapely import Polygon, MultiPolygon
from argparse import ArgumentParser

# Internal dependencies:
from lib.plotutils import map_polygons
from lib.ioutils import (ParameterSet,
                         load_points,
                         minmax_in_polygon_file,
                         change_zvalue_in_polygon_file,
                         load_polygons,
                         polygons_to_file)


if __name__ == "__main__":
    
    # Read input arguments:
    parser = ArgumentParser(description="Draw Voronoi diagrams or density maps from GMT ASCII tables")
    parser.add_argument("configfile", 
                        help="Configuration file")

    parser.add_argument("files",
                        help="Grids or Polygons formatted as GMT Multiple Segment Files",
                        nargs='+',
                        type=str)

    parser.add_argument("-c", "--colormap", 
                        help="Specify GMT colormap", 
                        default="roma")

    parser.add_argument("-i", "--invert",
                        help="Invert colormap",
                        action='store_true')

    parser.add_argument("-r", "--range", 
                        help="Specify lower and upper limits for the colormap", 
                        nargs=2,
                        type=float,
                        default=None)
                        
    parser.add_argument("-t", "--title",
                        help="Figure title")

    parser.add_argument('-b', '--boundaries',
                        help="Plot polygons boundaries",
                        action='store_true')

    parser.add_argument('-o', '--output-directory',
                        help="Output directory for figures",
                        default=None)

    parser.add_argument('-l', '--logvalues',
                        help="Apply log10 transformation to Z-values before plotting (linear colorscaling)",
                        action='store_true')

    parser.add_argument('-L', '--log-colorscale',
                        help="Map the colorscale to Z-values using a logarithmic binning scheme",
                        action='store_true')

    parser.add_argument('-n', '--no-coastlines-and-ticks',
                        help="Remove coastlines and axes ticks (useful for synthetics)",
                        action='store_true')

    parser.add_argument('-e', '--exclude-infinite',
                        help="Replace infinite Z-values by NaN to disable coloring. " +
                             "When option '-l' is used, replacement occurs AFTER log10 transformation",
                        action='store_true')

    parser.add_argument('-s', '--set-transparency',
                        help="Set transparency index, from 0 (opaque) to 100",
                        type=int,
                        default=30)

    parser.add_argument('-z', '--exclude-zeros',
                        help="Exclude cells with null Z-values. " +
                             "When option '-l' is used, exclusion occurs BEFORE log10 transformation",
                        action='store_true')

    parser.add_argument('--tight',
                        help="Tighten map boundaries to the bounding polygon defined in file 'bounds.txt'",
                        action='store_true')

    parser.add_argument('-p', '--projection',
                        help="Specify map projection using the GMT software single-letter code (default: M for Mercator)",
                        default='M')

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
    bounding_box = load_points(prms.bounds_file)
           
    for inputfile in args.files:
        print(inputfile)
        prefix, ext = os.path.splitext(os.path.basename(inputfile))
        tmpfiles = []

        # Exclude cells with null Z-values:
        if args.exclude_zeros:
            print('>> Exclude polygons with zero Z-values')
            # Duplicate polygon file:
            tmpfiles.append(prefix + "_z.tmp")
            mpoly, zvalues = load_polygons(inputfile)
            in0 = (zvalues != 0.0).nonzero()[0]
            zvalues = zvalues[in0]
            mpoly = MultiPolygon([mpoly.geoms[i] for i in in0])
            polygons_to_file(tmpfiles[-1], mpoly, zvalues)
            inputfile = tmpfiles[-1]

        # Apply log-transformation to Z-values:
        if args.logvalues:
            print('>> Compute log (base 10) of polygon Z-values')
            # Duplicate polygon file:
            tmpfiles.append(prefix + "_l.tmp")
            _, zvalues = load_polygons(inputfile)
            change_zvalue_in_polygon_file(inputfile, tmpfiles[-1], np.log10(zvalues))
            inputfile = tmpfiles[-1]

        # Replace infinite Z-values by NaN:
        if args.exclude_infinite:
            print('>> Exclude non-finite Z-values from the colorscale (set as NaN)')
            _, zvalues = load_polygons(inputfile)
            zvalues[np.logical_not(np.isfinite(zvalues))] = np.nan
            tmpfiles.append(prefix + "_e.tmp")
            change_zvalue_in_polygon_file(inputfile, tmpfiles[-1], zvalues)
            inputfile = tmpfiles[-1]

        # Load colormap limits:
        if args.range is None:
            cmin, cmax = minmax_in_polygon_file(inputfile, exclude_inf=True)
        else:
            cmin = args.range[0]
            cmax = args.range[1]
        print(f'>> Range of Z-values: [{cmin:.3g}; {cmax:.3g}]')
        
        # Set title:
        if args.title is None:
            title = prefix
        else:
            title = args.title

        # Plot polygons boundaries:
        if args.boundaries is True:
            pen_spec = "0.5p,gray,solid"
        else:
            pen_spec = None

        if args.no_coastlines_and_ticks:
            print('>> Removes coastlines and disables axes ticks')
            coast_res = None
            frame = 'f'
        else:
            coast_res = 'i'
            frame = 'af'

        if args.tight:
            print('>> Tighten map boundaries to the bounding box defined in bounds.txt')
            boundaries = bounding_box
            buffer = 0.0
        else:
            boundaries = None
            buffer = 0.0

        map_polygons(f'{inputfile}', 
                     clim=[cmin, cmax],
                     bounds=boundaries,
                     pen=pen_spec,
                     colormap=f"{args.colormap}",
                     colormap_reversal=args.invert,
                     colormap_nbins=30,
                     cbar_title=f"{title}",
                     savefig=True,
                     filename=os.path.join(args.output_directory, f'{prefix}.png'),
                     showfig=False,
                     coast_resolution=coast_res,
                     title=None,
                     dpi=300,
                     spatial_buffer=buffer,
                     add_polygon=bounding_box,
                     map_projection=f"{args.projection}15c",
                     logscale=args.log_colorscale,
                     figframe=frame,
                     transparency_index=args.set_transparency)

        # Delete temporary files:
        if len(tmpfiles) > 0:
            for file in tmpfiles:
                os.remove(file)
    
    
    