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

### Output files ###









 * 
 ![](/path/to/image.png)

*





> [!CAUTION]
> Use this program at your own risks! We cannot guarantee the exacteness and the reliability of any program included in this repository.

[^1]: Gutenberg, B., and Richter, C. F., 1944, Frequency of Earthquakes in California, _Bulletin of the Seismological Society of America_, 34, 4, pp.185-188.
