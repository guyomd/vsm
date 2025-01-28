# VSM: Voronoi-based Seismicity Models
Computer-programs for the construction of seismicity models and maps based on Voronoi diagrams.

>[!WARNING]
>:warning: ***Documentation under-construction...***

> [!NOTE]
> For citation and a detailed presentation of the method, see:\
> **Daniel, G., and Arroucau, P., 2025, Data-driven seismicity models based on Voronoi diagrams, _in preparation_**

## Package contents ##
This package consists in two main modules added of a suite of utilities for plotting or validation purposes:\
* **`voronoi_smoothing.py`**: Calculation of earthquake count/density grids based on Voronoi diagrams of a set of epicentral locations. This program can also propagate earthquake location and magnitude uncertainties using a random Monte-Carlo sampling process. This tends to produce a data-driven smoothing of spatial seismicity patterns. One count/density grid is constructed for each magnitude bin and is then projected onto a regular spatial mesh.
* **`compute_ab_values.py`**: Modelling of Frequency-Magnitude Distributions (FMD). This program uses a collection of count/density grids over several magnitude bins to reconstruct a FMD for each pixel of the regular spatial mesh. The program also estimates and returns a grid of parameters _a_ and _b_ from the Gutenberg & Richter law[^1] in each pixel.
* <ins>Plotting utilities</ins>:
  * **`plot_fmd.py`**: Plot the Frequency-Magnitude Distribution for one, or several, pixels
  * **`plot_map.py`**: Plot a map of earthquake counts/densities or _a_/_b_-values of the Gutenberg & Richter relationship
  * **`plot_distrib.py`**: Plot for one pixel of the regular grid the distribution of any quantity stored in GMT-formatted multiple segment files (_i.e.,_ counts, densities, _a_ or _b_). This can be useful to inspect the distributions produced by the propagation of uncertainties using `voronoi_smoothing.py`.  
    
* <ins>Testing utilities</ins>:
  * **`gof_tests.py`** : ***[ Under development ].*** Goodness-of-fit tests based on residual analysis.

## Examples ##
Tutorials and examples will be included to assist interested users with our input formats, and with the use of the programs.

## File formats ##

### Input files ###
* **Configuration file** (_e.g.,_ `parameters.txt`): Define the list of input files and settings for each application. Each line starting with "#" is considered as a comment and skipped.
  ```
  # VSM CONFIGURATION FILE
  # -- Define input files and output directories --
  file_for_epicenters: epicenters_synthetics.txt
  file_for_geographical_bounds: bounds.txt
  file_for_magnitude_bins: bins.txt
  #file_for_FMD_limits_and_durations: fmd_info.txt
  #file_for_prior_b_information: b_prior_info.txt
  output_directory_for_files: results
  output_directory_for_figures: figures
  # -- Specification of input CRS and internal CRS used for the computation of areas --
  input_CRS: EPSG:4326 
  # NB: EPSG:4326 --> WGS-84
  internal_equal_area_CRS: EPSG:2154  
  # NB: EPSG:2154 --> Lambert 93
  unit_for_internal_CRS_coordinates: m
  # -- Calculation settings --
  mesh_discretization_step: 0.5 deg
  # NB: available units for step: km, deg
  density_scaling_factor: 1000.0
  skip_ab_if_missing_priors: True
  define_completeness_automatically: False
  enable_verbosity: True
  # -- Options governing the propagation of uncertainties --
  nb_bootstrap_samples: 100
  perturb_magnitudes: True
  save_bootstrap_realizations: False
  nb_parallel_tasks: 3
  b_value_to_remove_bias_on_perturbed_magnitudes: 1.0
  save_bootstrap_realizations: False
    ```
* **Magnitude bin information file** (_e.g.,_ `bins.txt`): Definition of magnitude bins.\
  One bin per line, with columns arranged in the following order: `[ID] [MIN] [MAX] [TMIN] [TMAX]`. Each line starting with "#" is considered as a comment and skipped.
  ```
  # VSM MAGNITUDE BIN CONFIGURATION
  # [ID] [MIN] [MAX] [TMIN] [TMAX]
  1	2.0	2.5	0.0	10000.0
  2	2.5	3.0	0.0	10000.0
  3	3.0	3.5	0.0	10000.0
  4	3.5	4.0	0.0	10000.0
  5	4.0	4.5	0.0	10000.0
  6	4.5	5.0	0.0	10000.0
  7	5.0	5.5	0.0	10000.0
  8	5.5	6.0	0.0	10000.0
  9	6.0	6.5	0.0	10000.0
  10	6.5	7.0	0.0	10000.0
  ```
  
* **Target geographical area** (_e.g.,_ `bounds.txt`): Define the target geographical area.\
  Bounds can be specified in two forms:
    - rectangular area, or
    - any area enclosed in a specified polygon
  Each line starting with "#" is considered as a comment and skipped.
  
  <ins>Example format for a rectangular area</ins>: \
  It is defined by its 4 corners, with columns arranged in `[LON] [LAT]` order. 
  ```
  # VSM TARGET GEOGRAPHICAL AREA
  -7.25  41.75
  11.25  41.75
  11.25  51.25
  -7.25  51.25
  ```

  <ins>Example format for an area enclosed in a polygon</ins>: \
  It is defined by its 4 corners, with columns arranged in `[LON] [LAT]` order. The example below represents the polygon enclosing the Californian area used in the RELM[^2] earthquake forecasting experiment.
  ```
  # VSM TARGET GEOGRAPHICAL AREA
  -125.4  43.0
  -118.16447368421052  43.0
  -113.4  35.9
  -113.1  35.45
  -113.1  32.18333333333337
  -113.2  31.8
  -113.3  31.7
  -113.5  31.6
  -113.7  31.5
  -118.36842105263159  31.5
  -121.40  33.3
  -121.5  33.4
  -121.7  33.6
  -122.0  33.9
  -122.2  34.2
  -123.2  35.7
  -124.7  38.0
  -125.4  39.1
  -125.4  39.1
  -125.4  43.0
  ```
* **Earthquake catalogue** (_e.g.,_ `epicenters.txt`): Catalogue of earthquake epicentral locations with location and magnitude uncertainties.\
  One epicenter per line. Each line starting with "#" is considered as a comment and skipped.\
  Two formats are available, depending on whether uncertainties are included in the catalogue, or not:
  - <ins>without uncertainties</ins>, columns are arranged in the following order: `[FLOATING DATE] [LON] [LAT] [MAG]`
  - <ins>with uncertainties</ins>, columns are arranged in the following order: `[FLOATING DATE] [LON] [LAT] [MAG] [HALF-LENGTH OF SEMI-MAJOR AXIS IN KM] [HALF-LENGTH OF SEMI-MINOR AXIS IN KM] [AZIMUTH OF SEMI-MAJOR AXIS in DEG] [MAG UNCERTAINTY]`
  ```
  # VSM EARTHQUAKE EPICENTERS 
  # 1 event per line
  # Coordinates expressed in geographical coordinates (EPSG:4326)	
  # Line format: Floating_Date  Longitude  Latitude  Magnitude  Loc_Unc_SMAJ  Loc_Unc_SMIN  Loc_Unc_Az  Mag_Unc
  1522.479452	0.75	46.917	5.8	100.0	100.0	0.0	0.43
  1579.068493	2.0	46.583	6.0	50.0	50.0	0.0	0.47
  1580.262295	1.5	51.0	5.8	50.0	50.0	0.0	0.49
  1618.501370	-0.617	43.2	5.1	50.0	50.0	0.0	0.48
  1640.510929	-1.367	48.933	5.2	100.0	100.0	0.0	0.42
  1650.720548	7.6	47.55	5.3	50.0	50.0	0.0	0.44
  1657.123288	0.617	47.117	5.7	50.0	50.0	0.0	0.51
  1663.035616	-0.75	46.95	5.3	100.0	100.0	0.0	0.5
  1665.093151	-0.05	43.1	5.4	100.0	100.0	0.0	0.43
  1672.945355	7.75	47.483	5.6	50.0	50.0	0.0	0.45
  1678.668493	5.783	43.75	5.0	100.0	100.0	0.0	0.4
  1682.358904	6.517	47.967	6.3	20.0	20.0	0.0	0.37
  1704.191257	0.75	46.95	5.3	100.0	100.0	0.0	0.51
  1706.682192	0.4	47.317	4.9	50.0	50.0	0.0	0.45
  1708.196721	0.017	47.05	5.3	100.0	100.0	0.0	0.38
  1711.761644	0.05	46.933	5.7	10.0	10.0	0.0	0.37
  1728.587432	7.917	48.35	6.0	10.0	10.0	0.0	0.33
  1736.445355	7.333	47.383	4.5	100.0	100.0	0.0	0.53
  1737.375342	8.3	48.917	5.3	50.0	50.0	0.0	0.5
  1743.178082	-0.75	43.25	5.6	100.0	100.0	0.0	0.66
  1749.775342	0.75	46.567	4.9	50.0	50.0	0.0	0.34
  1750.391781	-0.033	43.067	6.1	10.0	10.0	0.0	0.38
  1752.125683	7.733	43.8	4.5	100.0	100.0	0.0	0.52
  1755.098630	-2.183	48.2	4.5	100.0	100.0	0.0	0.44
  1755.986301	6.311	50.861	5.09	12.3	9.6	0.0	0.37
  1756.090164	-0.633	46.033	4.7	25.0	25.0	0.0	0.44
  ```
   
* **[Optional] FMD properties for each pixel** (_e.g.,_ `fmd_info.txt`):
* **[Optional] Prior _b_-value information for each pixel** (_e.g.,_ `b_prior_info.txt`):

### Output files ###









 * 
 ![](/path/to/image.png)

*





> [!CAUTION]
> Use this program at your own risks! We cannot guarantee the exacteness and the reliability of any program included in this repository.

[^1]: Gutenberg, B., and Richter, C. F., 1944, Frequency of Earthquakes in California, _Bulletin of the Seismological Society of America_, 34, 4, pp.185-188.
[^2]: Field, E. H., 2007, Overview of the Working Group for the Development of Regional Earthquake Likelihood Models (RELM), Seismological Research Letters, 78, 1, pp.7-16
