import numpy as np
from pathlib import Path
from shapely import intersection
import pygmt 
from tqdm import tqdm

from vsm.lib.ioutils import (load_points, 
                             load_polygons, 
                             load_grid, 
                             load_bins, 
                             load_fmd_file,
                             ParameterSet)
from vsm.lib.geoutils import convert_to_EPSG


CONFIGFILE = Path('..') / '..' / 'AreaSources' / 'BRGM_gk74' / 'parameters.txt' 
#CONFIGFILE = Path('..') / '..' / 'AreaSources' / 'IRSN_gk74' / 'parameters.txt' 


### Load configuration and results: ###
VSMDIR = CONFIGFILE.parent
params = ParameterSet()
params.load_settings(CONFIGFILE)

# Load a-values scale to each polygon area
cells, a_values = load_polygons(VSMDIR / 'results' / 'a_cells_bootstrap.txt')
# Load b-values:
_, b_values = load_polygons(VSMDIR / 'results' / 'b_cells_bootstrap.txt')
# cells, a_values = load_polygons(VSMDIR / 'results' / 'a_cells.txt')
# _, b_values = load_polygons(VSMDIR / 'results' / 'b_cells.txt')
# cells, a_values = load_polygons(VSMDIR / 'results' / 'a_cells_aggregated.txt')
# _, b_values = load_polygons(VSMDIR / 'results' / 'b_cells_aggregated.txt')

# Load events from input catalogue:
points, dates, mags, weights, uncertainties = load_points(VSMDIR / params.epicenters_file)

# Load bounding box of the whole study area:
if params.bounds_file is None:
    bounding_box = cells.envelope
else:
    bounding_box = load_points(VSMDIR / params.bounds_file)

### Compute bin-wise observed and modelled rates: ###
bins = load_bins(VSMDIR / params.bins_file)  # Each element is a 4-element list 
densities, ncells, nbins, bin_ids = load_grid(VSMDIR / 'results' / 'gridded_densities.txt', 
                                              scaling_factor=1.0)
lons = densities[:, 0]
lats = densities[:, 1]
cellinfo, bin_durations = load_fmd_file(VSMDIR / params.bins_file,
                                        lons,
                                        lats,
                                        str(VSMDIR / params.fmd_info_file),
                                        ibins=[int(v - 1) for v in bins[:, 0]],
                                        mmin=None,
                                        mmax=None,
                                        coord_precision=1E-6,
                                        verbose=True)

total_rate_model = np.zeros((nbins,))  # Sum of cell-wise modelled rates
total_rate_obs = np.zeros((nbins,))  # Sum of cell-wise observed rates
total_catalog_events = np.zeros((nbins))
ratios = np.zeros((ncells,))
for i in tqdm(range(ncells)):
    c = cells.geoms[i]
    a = a_values[i]
    b = b_values[i]
    inter = intersection(c, bounding_box)
    
    # Convert coordinates using metric EPSG:
    c_converted = convert_to_EPSG(c, params.input_epsg, params.internal_epsg)  
    inter_converted = convert_to_EPSG(inter, params.input_epsg, params.internal_epsg)
    ratios[i] = inter_converted.area / c_converted.area
    cell_area_km2 = c_converted.area * (params.epsg_scaling2km ** 2)  
    for j in range(nbins):
        m1 = bins[j, 1]
        m2 = bins[j, 2]
        tmax = bins[j, 4]
        comp_dur = bin_durations[i, j] 
        # Modelled seismicity rates (per year) for each bin and for the whole cell area:
        total_rate_model[j] += ratios[i] * ( 10 ** (a - b * m1) - 10 ** (a - b * m2)) 
        # Estimated seismicity rates (per year):
        total_rate_obs[j] += ratios[i] * (densities[i, 2 + j] / params.density_scaling_factor * cell_area_km2) \
                                      / comp_dur
        # Count events included in polygon, in magnitude bin and in completeness period:
        counts = 0
        for k in range(len(points.geoms)):
            #if points.geoms[k].within(c) and (mags[k] >= m1) and (mags[k] < m2) and (dates[k] >= (tmax - comp_dur)):
            if points.geoms[k].within(c) and (mags[k] >= m1) and (mags[k] < m2):
                counts += 1
        total_catalog_events[j] += counts / comp_dur
 



print('Obs')
print(total_rate_obs)
print('Model')
print(total_rate_model)
print('Catalogue')
print(total_catalog_events)

# Build inverse cumulative laws:
cumul_total_rate_model = np.flip(np.flip(total_rate_model).cumsum())
cumul_total_rate_obs = np.flip(np.flip(total_rate_obs).cumsum())
cumul_total_catalog_events = np.flip(np.flip(total_catalog_events).cumsum()) 

# Make plot:
mags = bins[:, 1] + 0.5 * (bins[:, 2] - bins[:, 1])
ymin = 10 ** np.floor(np.log10(min([cumul_total_rate_model.min(), cumul_total_rate_obs.min()])))
ymax = 10 ** np.ceil(np.log10(max([cumul_total_rate_model.max(), cumul_total_rate_obs.max()])))

fig = pygmt.Figure()
fig.basemap(region=[bins[:, 1].min(), bins[:, 2].max(), ymin, ymax],
            projection="X10c/10cl",
            frame=['WSne+tWhole area', 'xafg+lMagnitude', 'yafg3+lSeismicity Rate'])
fig.plot(x = mags,
         y = cumul_total_catalog_events,
         style='s0.3c',
         fill='black',
         label='Total catalogue rates'
         )
fig.plot(x = mags,
         y = cumul_total_catalog_events,
         pen='1p,black,solid',
         )
fig.plot(x = mags,
         y = cumul_total_rate_obs,
         style='s0.3c',
         fill='steelblue1',
         label='Total estimated rate'
         )
fig.plot(x = mags,
         y = cumul_total_rate_obs,
         pen='1p,steelblue1,solid'
         )
fig.plot(x = mags,
         y = cumul_total_rate_model,
         style='s0.3c',
         fill='orange',
         label='Total modelled rate'
         )
fig.plot(x = mags,
         y = cumul_total_rate_model,
         pen='1p,orange,solid',
         )
fig.legend()
fig.savefig(VSMDIR / 'QC_total_counts.png')


