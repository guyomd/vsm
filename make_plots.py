"""
Draw plots of Voronoi polygons and gridded annual counts for each magnitude bin

SYNTAX:
- To plot all figures:
    "python3 make_plots.py"
or, "python3 make_plots.py all"

- To plot only specific figures:
    "python3 make_plots.py voronoi_polygons"
or, "python3 make_plots.py annual_counts"
or, "python3 make_plots.py voronoi_polygons abvalues"


"""
import os
import glob
import sys
from shapely import Polygon, Point
import numpy as np

# Internal dependencies:
from lib.plotutils import map_polygons, plot_mfd
from lib.ioutils import ParameterSet, load_points, minmax_in_polygon_file, load_bins


def _set_value_if_not_nan(value, additional_condition=True):
    if np.isnan(value):
        return None
    elif additional_condition:
        return value


if __name__ == "__main__":
    
    
    # Syntax: 
    # python make_plots.py parameters.txt [voronoi_polygons] [ab_values] [density_maps] [all] [histo_1] ... [histo_2]
    
    # Initialization:
    do_plot_voronoi_polygons = False
    do_plot_density_maps = False
    do_plot_ab_values = False
    do_plot_fmd_histogram = False
    
    # Load parameters:
    prms = ParameterSet()
    prms.load_settings(sys.argv[1])
            
    # Check output directory existence:
    if not os.path.isdir(prms.figures_dir):
        os.mkdir(prms.figures_dir)

    envelope = load_points(prms.bounds_file)
           
    # Read optional command-line arguments:
    opts = sys.argv[2:]
    histo_ids = []
    if len(opts) > 0:
        for arg in opts:
            if (arg == "voronoi_polygons") or (arg == "polygons"):
                do_plot_voronoi_polygons = True
            elif (arg == "density_maps") or (arg == "density"):
                do_plot_density_maps = True
            elif arg == "ab_values":
                do_plot_ab_values = True
            elif arg.startswith('histo_'):  # e.g. 'histo_1' or 'histo_2'
                do_plot_fmd_histogram = True
                histo_ids.append(int(arg.split('_')[1]))
            elif arg == "all":
                do_plot_voronoi_polygons = True
                do_plot_density_maps = True
                do_plot_ab_values = True
    else:
        print('>> No optional arguments specified.\n   Will plot polygons and maps for density and (a, b) values.')
        do_plot_voronoi_polygons = True
        do_plot_density_maps = True
        do_plot_ab_values = True

    if do_plot_voronoi_polygons:
        for polfile in glob.glob(os.path.join(prms.output_dir, 'polygons_bin_*.txt')):
            print(polfile)
            prefix, ext = os.path.splitext(os.path.basename(polfile))
            map_polygons(f'{polfile}', 
                        clim=[-3, 3],
                        #clim=[-9, -4],
                        bounds = None,
                        colormap='red2green', 
                        colormap_reversal=True, 
                        colormap_nbins=30, 
                        cbar_title = f"Earthquake density per year (log@-10@-)",
                        savefig=True, 
                        filename=os.path.join(prms.figures_dir, f'{prefix}.png'), 
                        showfig=False, 
                        coast_resolution="i", 
                        title=None, 
                        dpi=300,
                        spatial_buffer=0.0,
                        add_polygon=envelope)

    if do_plot_density_maps:        
        for densityfile in glob.glob(os.path.join(prms.output_dir, 'annual_density_bin*.txt')):
            print(densityfile)       
            prefix, ext = os.path.splitext(os.path.basename(densityfile)) 
            map_polygons(f'{densityfile}', 
                        clim=[-3, 3],
                        #clim=[-6, 3],
                        bounds = None,
                        pen = None,
                        colormap='red2green', 
                        colormap_reversal=True, 
                        colormap_nbins=30, 
                        cbar_title = f"Annual earthquake density, per km@+2@+ (log@-10@-)",
                        savefig=True, 
                        filename=os.path.join(prms.figures_dir, f'{prefix}.png'), 
                        showfig=False, 
                        coast_resolution="i", 
                        title=None, 
                        dpi=300,
                        spatial_buffer=0.0,
                        add_polygon=envelope,
                        map_projection="M15c")

    if do_plot_ab_values:
        grt_files = {os.path.join(prms.output_dir, 'a_cells.txt'): 
                         [f'a (per {prms.density_scaling_factor} km@+2@+)', 'red2green', True],
                     os.path.join(prms.output_dir, 'b_cells.txt'): 
                         ['b', 'copper', False, [0.8, 1.2]],
                     os.path.join(prms.output_dir, 'rho_ab_cells.txt'): 
                         ['@~\\162@~@-ab@-', 'polar', True, [0.8, 1.0]],
                     os.path.join(prms.output_dir, 'da_cells.txt'): 
                         ['@~\\144@~a', 'polar', False ],
                     os.path.join(prms.output_dir, 'db_cells.txt'):
                         ['@~\\144@~b', 'polar', False,  [1E-5, 1E-3]], 
                     os.path.join(prms.output_dir, 'mmax_cells.txt'): 
                         ['M@-max@-', 'red2green', True ]
                     }
        for resfile in grt_files.keys():
            print(resfile)       
            prefix, ext = os.path.splitext(os.path.basename(resfile))
            if len(grt_files[resfile]) < 4:
                zmin, zmax = minmax_in_polygon_file(resfile)
            else:
                zmin, zmax = grt_files[resfile][3]
            print(f'Range: [min, max] = [{zmin}, {zmax}]') 
            map_polygons(f'{resfile}', 
                        clim=[zmin, zmax],
                        bounds = None,
                        pen = None,
                        colormap=grt_files[resfile][1], 
                        colormap_reversal=grt_files[resfile][2], 
                        colormap_nbins=30, 
                        cbar_title = f"{grt_files[resfile][0]}",
                        savefig=True, 
                        filename=os.path.join(prms.figures_dir, f'{prefix}.png'), 
                        showfig=False, 
                        coast_resolution="i", 
                        title=None, 
                        dpi=300,
                        spatial_buffer=0.0,
                        add_polygon=envelope,
                        map_projection="M15c")
    
    
    if do_plot_fmd_histogram:
        
        # Load seismicity rates and truncated G-R parameters:
        all_rates = np.loadtxt(os.path.join(prms.output_dir, 'gridded_rates.txt'), delimiter=';')
        #all_rates[:, 2:] *= prms.density_scaling_factor # Scale rates
        bins = load_bins(prms.bins_file)
        minmags = bins[:, 1]
        maxmags = bins[:, 2]      
        grt_prms = np.loadtxt(os.path.join(prms.output_dir, 'ab_values.txt'), delimiter=';')
        if prms.fmd_info_file:
            fmd_info = np.loadtxt(prms.fmd_info_file, delimiter=';')
        else:
            fmd_info = None
        
        # Plot FMD for all pixels requested:    
        for indx in histo_ids:
            centroid = Point([all_rates[indx, 0], all_rates[indx, 1]])
            print(f">> Plot MFD for pixel {indx} ({centroid.x}, {centroid.y})")
            a = _set_value_if_not_nan(grt_prms[indx, 2])
            b = _set_value_if_not_nan(grt_prms[indx, 3])
            da = _set_value_if_not_nan(grt_prms[indx, 4])
            if fmd_info is not None:
                mmax = _set_value_if_not_nan(fmd_info[indx, 3], additional_condition=lambda x: x > 0)
            plot_mfd(minmags, maxmags, all_rates[indx, 2:], envelope, centroid, 
                     a=a, 
                     b=b,
                     da=da,
                     show_inset=True, 
                     mmax=mmax, 
                     savefig=os.path.join(prms.figures_dir, f"mfd_pixel_{indx}.png"))
        
            
