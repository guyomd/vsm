import sys
import os
import numpy as np
from shapely import MultiPoint, Polygon, Point
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

# Internal dependencies:
from lib.ioutils import ParameterSet, load_points, load_bins, polygons_to_file
from lib.geoutils import (convert_to_EPSG,
                          eqdensity_per_polygon,
                          clipped_voronoi_diagram,
                          build_mesh,
                          eqcounts_per_cell,
                          select_events,
                          interpolate_polygon_coords,
                          random_locations_from_ellipsoid,
                          subdivide_voronoi_cells,
                          reorder_germs)


class VoronoiSmoothingAlgorithm:

    def __init__(self, configfile: str = None):
        self.prms = ParameterSet()
        if configfile:
            self.prms.load_settings(configfile)

    def load_input_data(self):
        mp_epic, dates, mags, uncert = load_points(self.prms.epicenters_file)
        bounds = load_points(self.prms.bounds_file)

        # Defensive programming: Check if uncertainties provided, when needed:
        if (self.prms.nb_bootstrap_samples > 0) and (uncert is None):
            raise ValueError('Bootstrap required: missing uncertainties in "{self.prms.epicenters_file}"')

        # Discretize bounding polygon more finely:
        bounds = interpolate_polygon_coords(bounds, n=1000)

        # Convert coordinates to metric units:
        mp_epic_m = convert_to_EPSG(mp_epic,
                                    in_epsg=self.prms.input_epsg,
                                    out_epsg=self.prms.internal_epsg)
        bounds_m = convert_to_EPSG(bounds,
                                   in_epsg=self.prms.input_epsg,
                                   out_epsg=self.prms.internal_epsg)

        magbins = load_bins(self.prms.bins_file)
        return mp_epic, mp_epic_m, dates, mags, uncert, bounds, bounds_m, magbins

    def build_regular_mesh(self, mesh_step_unit, bounds, bounds_m):
        if mesh_step_unit == "km":
            cells_m, centroids_m = build_mesh(bounds_m,
                                              self.prms.mesh_step,
                                              scaling2unit=1 / self.prms.epsg_scaling2km)
            cells = convert_to_EPSG(cells_m,
                                    in_epsg=self.prms.internal_epsg,
                                    out_epsg=self.prms.input_epsg)
            centroids = np.array([(c.centroid.x, c.centroid.y) for c in cells.geoms])

        elif mesh_step_unit == "deg":
            cells, centroids = build_mesh(bounds,
                                          self.prms.mesh_step,
                                          scaling2unit=1.0)
            cells_m = convert_to_EPSG(cells,
                                      in_epsg=self.prms.input_epsg,
                                      out_epsg=self.prms.internal_epsg)

        else:
            raise ValueError(f'Incorrect mesh step unit: "{mesh_step_unit}"')

        return cells, cells_m, centroids

    def reset_values_for_cells_beyond_bounds(self, cells, bounds, *args):
        """
        Set a null value to cells with centroid located beyond the bounding polygon

        :param cells:
        :param bounds:
        :return:
        """
        if len(args) < 1:
            raise SyntaxError('Input arguments must contain at least one array of values')
        for i in range(len(cells.geoms)):
            if cells.geoms[i].centroid.within(bounds) is False:
                for arg in args:  # Loop over all optional input arguments provided (1 or more)
                    arg[i] = 0.0
        return args

    def density_grid_for_single_bin(self, magbin, mp_epic_m, mags, dates, bounds_m, cells_m,
                                    rng, uncert=None, bootstrap=False):
        """
            Core routines implementing the following tasks:
            1 - Select only events included in the magnitude bin range (and in the time period, to be removed)
            2 - Compute clipped Voronoi diagram for the current magnitude bin
            3 - Compute densities (in km**2) for each polygon
            4 - Compute densities in every cells using the fraction of polygons that intersects each cell
        """
        if bootstrap:
            verbose = False
        else:
            verbose = True
            perturbed_catalogue = None
        index = int(magbin[0])
        duration = magbin[4] - magbin[3]
        if verbose:
            print(f'>> bin {index}: M in [{magbin[1]}; {magbin[2]}[  ({duration} years)')

        # If requested, apply bootstrapping (NB: when requested, the perturbation of magnitudes occurs earlier):
        if bootstrap:
            mp_epic_m, mags, dates = self.bootstrap_catalogue_sample(mp_epic_m, mags, dates, uncert, rng)

        # Keep only events with magnitude included in the current bin, and located within the bounding box:
        mp_epic_bin, m_bin, t_bin = select_events(mp_epic_m, mags, dates, bounds_m, magbin)
        nev = len(mp_epic_bin.geoms)

        # Keep track of the perturbed locations (and also magnitudes, possibly) in a dedicated variable:
        if bootstrap:
            mp_epic_bin_input_epsg = convert_to_EPSG(mp_epic_bin,
                                                     in_epsg=self.prms.internal_epsg, 
                                                     out_epsg=self.prms.input_epsg)
            nev = len(mp_epic_bin.geoms)
            perturbed_catalogue = np.zeros((nev, 4))
            for i in range(nev):
                x, y = mp_epic_bin_input_epsg.geoms[i].coords.xy
                perturbed_catalogue[i, 0] = t_bin[i]  # Perturbed occurrence times
                perturbed_catalogue[i, 1] = x[0]      # Perturbed longitudes
                perturbed_catalogue[i, 2] = y[0]      # Perturbed latitudes
                perturbed_catalogue[i, 3] = m_bin[i]  # Perturbed magnitudes

                
        if verbose:
            print(f'>> bin {index}: {nev} epicenters selected')
        if nev == 0:
            if verbose:
                print(f">> bin {index}: Skip bin, requires at least 1 epicenter")
            return None

        # Compute clipped Voronoi diagram using the internal CRS:
        vor_diagram_m, weights = clipped_voronoi_diagram(mp_epic_bin,
                                                         bounds_m,
                                                         verbose=self.prms.is_verbose)

        # If requested, sub-divide Voronoi cells in sub-triangles:
        if self.prms.subdivide_polygons:
            germs = reorder_germs(vor_diagram_m, mp_epic_bin)
            vor_diagram_m, weights = subdivide_voronoi_cells(vor_diagram_m, weights, germs)

        # Compute polygon density per km2:
        vor_densities_km2 = eqdensity_per_polygon(vor_diagram_m,
                                                  weights,
                                                  scaling2unit=self.prms.epsg_scaling2km,
                                                  log_values=False)
        vor_densities_km2 *= self.prms.density_scaling_factor

        # Project Voronoi polygons in the input CRS:
        vor_diagram = convert_to_EPSG(vor_diagram_m,
                                      in_epsg=self.prms.internal_epsg,
                                      out_epsg=self.prms.input_epsg)

        # Compute earthquake counts and densities in each cell:
        counts, cell_densities_km2 = eqcounts_per_cell(cells_m,
                                                       vor_diagram_m,
                                                       weights,
                                                       scaling2unit=self.prms.epsg_scaling2km)

        if np.abs(np.round(counts.sum()) - weights.sum()) > 0.01:
            raise AssertionError(f'Total counts over the mesh domain ({counts.sum()}) differ of' +
                                 f'more than 0.01 from the total counts in polygons ({weights.sum()})')

        # Reset values for cells with centroids located beyond bounds:
        counts, cell_densities_km2 = self.reset_values_for_cells_beyond_bounds(cells_m,
                                                                               bounds_m,
                                                                         counts,
                                                                               cell_densities_km2)

        cell_densities_km2 *= self.prms.density_scaling_factor
        return counts, cell_densities_km2, vor_diagram, vor_densities_km2, perturbed_catalogue

    def write_output_for_GMT(self, directory, filename, cells, values, verbose=True):
        filepath = os.path.join(directory, filename)
        polygons_to_file(filepath, cells, zvalues=values, verbose=verbose)

    def write_matrix_CSV(self, directory, filename, values, columns_titles,
                         verbose=True, delimiter='; '):
        csvfile = os.path.join(directory, filename)
        np.savetxt(csvfile,
                   values,
                   header='; '.join(columns_titles),
                   delimiter=delimiter)
        if verbose:
            print(f'{csvfile}:: saved values for {values.shape[0]} cells')

    def perturb_magnitudes(self, mags, uncert: dict, rng, correct_bias=True, b_value=1.0):
        """

        :param mags: numpy.ndarray, array of original magnitude values
        :param uncert: dict, magnitudes uncertainties stored in field uncert['mag_unc']
        :param rng: numpy.random.Generator instance, pseudo-random number generator
        :param correct_bias: bool, specify whether, or not, to correct for the bias induced by a symmetrical random
            perturation of original magnitude values centered on their original values. When bias is not corrected,
            then, the final distribution of events contains a higher proportion of larger magnitude events. Note that
            we use a b-value of 1.0 to compute the correction term.
            See Dutfoy, A., 2023, Uncertainty on Estimatead magnitudes: A new approach based on a Poisson process of
                                  dimension 2, Pure and Applied Geophysics, 180, 919-923, section 4.7.1
        :param b_value: float, b-value of the Gutenerbg-Richter model used for the correction term. Default: 1.0
        :return: numpy.ndarray, array of randomly perturbed magnitude values
        """
        magp = np.array([rng.normal(loc=mags[i], scale=uncert['mag_unc'][i]) for i in range(len(mags))])
        if correct_bias:
            beta = b_value * np.log(10)
            magp -= 0.5 * beta * np.power(uncert['mag_unc'], 2)
        return magp

    def bootstrap_catalogue_sample(self, mp_epic_m, mags, dates, uncert: dict, rng):
        """
        Bootstrap sampling of 1 input catalogue sample, taking into account the Poisson-process variability in total
        events count and in location uncertainties.
        """
        nev0 = len(mp_epic_m.geoms)
        nev = rng.poisson(lam=nev0)  # Random Poisson variate for total events count
        indices = rng.choice(nev0, size=nev)
        # Do not perturb occurrence times and magnitudes
        # NB: Magnitudes already perturbed in method self.perturb_catalogue_mags():
        bs_dates = dates[indices]
        bs_mags = mags[indices]
        pts = []
        for i in indices:
            x_km, y_km = mp_epic_m.geoms[i].coords.xy
            x, y = random_locations_from_ellipsoid(x_km,
                                                   y_km,
                                                   uncert['loc_smaj'][i],
                                                   uncert['loc_smin'][i],
                                                   uncert['loc_az'][i],
                                                   n=1,
                                                   rng=rng)
            pts.append((x, y))
        bs_mp_epic_m = MultiPoint(pts)
        return bs_mp_epic_m, bs_mags, bs_dates

    def create_density_maps_for_all_bins(self, bs_index, magbins, mp_epic_m, mags, dates, bounds_m,
                                         cells, cells_m, uncert, counts, cell_densities_km2,
                                         suffix, outputdir, do_bootstrap_catalog, do_save_results):
        if do_bootstrap_catalog:
            verbose = False
        else:
            verbose = True
        rng = np.random.default_rng()  # Generator must be inside functions to ensure independent realizations
                                       # when multithreading is activated
        col_titles = ['lon', 'lat']
        col_index = 2
        pbar_linepos = 2 + np.mod(bs_index - 1, self.prms.nb_parallel_tasks)
        for magbin in tqdm(magbins, desc="Loop on bins", position=pbar_linepos, leave=False):
            bin_index = int(magbin[0])  # The magnitude-bin "id" field is given in column 0
            col_titles.append(f'bin_{bin_index:d}')
            outputs = self.density_grid_for_single_bin(magbin,
                                                       mp_epic_m,
                                                       mags,
                                                       dates,
                                                       bounds_m,
                                                       cells_m,
                                                       rng,
                                                       uncert=uncert,
                                                       bootstrap=do_bootstrap_catalog)
            if outputs is None:
                continue  # When no event in current bin
            else:
                counts[:, col_index] = outputs[0]
                cell_densities_km2[:, col_index] = outputs[1]
                vor_diag = outputs[2]
                vor_densities_km2 = outputs[3]
                perturbed_catalogue = outputs[4]

            if do_save_results:
                if perturbed_catalogue is not None:
                    # Write perturbed catalogue in CSV format:
                    colnames = ['Time', 
                                'Longitude', 
                                'Latitude', 
                                'Magnitude']
                    self.write_matrix_CSV(outputdir,
                                          f"catalog_bin_{bin_index}{suffix}.txt",
                                          perturbed_catalogue, 
                                          colnames,
                                          verbose=verbose,
                                          delimiter=' ')

                # Write output files in GMT format:
                self.write_output_for_GMT(outputdir,
                                          f"counts_bin_{bin_index}{suffix}.txt",
                                          cells,
                                          counts[:, col_index],
                                          verbose=verbose)
                self.write_output_for_GMT(outputdir,
                                          f"polygons_bin_{bin_index}{suffix}.txt",
                                          vor_diag,
                                          vor_densities_km2,
                                          verbose=verbose)
                self.write_output_for_GMT(outputdir,
                                          f"density_bin_{bin_index}{suffix}.txt",
                                          cells,
                                          cell_densities_km2[:, col_index],
                                          verbose=verbose)
            col_index += 1
        return bs_index, counts, cell_densities_km2, col_titles


    def run(self):
        # Load input data:
        mp_epic, mp_epic_m0, dates0, mags0, uncert, bounds, bounds_m, magbins = self.load_input_data()

        # Build zoneless mesh:
        cells, cells_m, centroids = self.build_regular_mesh(self.prms.mesh_step_unit,
                                                            bounds,
                                                            bounds_m)
        nbins = magbins.shape[0]
        ncells = centroids.shape[0]
        counts = np.zeros((ncells, nbins + 2))
        counts[:, 0:2] = centroids
        cell_densities_km2 = np.zeros((ncells, nbins + 2))
        cell_densities_km2[:, 0:2] = centroids
        outputdir = self.prms.output_dir
        suffix = ''

        # Initialize dates, locations and magnitudes to catalogue values:
        dates = dates0
        mp_epic_m = mp_epic_m0
        mags = mags0

        # Loop over magnitude bins (and boostrap realizations, if requested):
        if self.prms.nb_bootstrap_samples == 0:
            _, counts, cell_densities_km2, col_titles = self.create_density_maps_for_all_bins(
                0, magbins, mp_epic_m, mags, dates, bounds_m, cells, cells_m, uncert,
                counts, cell_densities_km2, suffix, outputdir, False, True)

        elif self.prms.nb_bootstrap_samples > 0:
            bs_counts = np.tile(counts, (self.prms.nb_bootstrap_samples, 1, 1))
            bs_cell_densities_km2 = np.tile(cell_densities_km2,
                                            (self.prms.nb_bootstrap_samples, 1, 1))
            bs_nz = np.floor(np.log10(self.prms.nb_bootstrap_samples) + 1.0).astype(int)
            outputdir = os.path.join(self.prms.output_dir, 'bootstrap')
            # Use parallelization:
            if self.prms.nb_parallel_tasks is None:
                ntasks = min(cpu_count() - 1, self.prms.nb_bootstrap_samples)
            else:
                ntasks = self.prms.nb_parallel_tasks
            print(f'>> Number of parallel processes: {ntasks}')
            with Pool(ntasks) as p:
                args = []
                rng = np.random.default_rng()
                for i in range(self.prms.nb_bootstrap_samples):
                    if self.prms.perturb_magnitudes:
                        mags = self.perturb_magnitudes(mags0,
                                                       uncert,
                                                       rng,
                                                       correct_bias=True,
                                                       b_value=self.prms.b_value_for_correction_term)
                    suffix = f'_bs_{i + 1:0{bs_nz}d}'
                    args.append([i + 1, magbins, mp_epic_m, mags, dates, bounds_m, cells, cells_m,
                                 uncert, counts, cell_densities_km2, suffix, outputdir, True,
                                 self.prms.save_realizations])

                for result in p.starmap(self.create_density_maps_for_all_bins,
                                        tqdm(args,
                                             position=1,
                                             desc='Bootstrapping...',
                                             leave=True),
                                        chunksize=1):
                    bs_index = result[0] - 1
                    bs_counts[bs_index, :, :] = result[1]
                    bs_cell_densities_km2[bs_index, :, :] = result[2]
                    col_titles = result[3]
                    if self.prms.save_realizations:
                        print('\n')  # for pretty display purposes...
                        # Write matrices of counts and annual densities (per km^2) for all bins:
                        self.write_matrix_CSV(outputdir,
                                              f'gridded_counts{suffix}.txt',
                                              result[1],
                                              col_titles,
                                              verbose=False)
                        self.write_matrix_CSV(outputdir,
                                              f'gridded_densities{suffix}.txt',
                                              result[2],
                                              col_titles,
                                              verbose=False)
            counts = bs_counts.mean(axis=0)
            counts_std = bs_counts.std(axis=0)
            cell_densities_km2 = bs_cell_densities_km2.mean(axis=0)
            cell_densities_std = bs_cell_densities_km2.std(axis=0)

            # Write ASCII files for std deviations of counts/densities:
            self.write_matrix_CSV(self.prms.output_dir,
                                  f'gridded_counts_std.txt',
                                  counts_std,
                                  col_titles)
            self.write_matrix_CSV(self.prms.output_dir,
                                  f'gridded_densities_std.txt',
                                  cell_densities_std,
                                  col_titles)

            # Write ASCII files for GMT for bin-wise average and std counts/densities:
            for bin_index in range(nbins):
                self.write_output_for_GMT(self.prms.output_dir,
                                          f"counts_bin_{bin_index + 1}.txt",
                                          cells,
                                          counts[:, bin_index + 2],
                                          verbose=True)
                self.write_output_for_GMT(self.prms.output_dir,
                                          f"counts_std_bin_{bin_index + 1}.txt",
                                          cells,
                                          counts_std[:, bin_index + 2],
                                          verbose=True)
                self.write_output_for_GMT(self.prms.output_dir,
                                          f"density_bin_{bin_index + 1}.txt",
                                          cells,
                                          cell_densities_km2[:, bin_index + 2],
                                          verbose=True)

                self.write_output_for_GMT(self.prms.output_dir,
                                          f"density_std_bin_{bin_index + 1}.txt",
                                          cells,
                                          cell_densities_std[:, bin_index + 2],
                                          verbose=True)

        self.write_matrix_CSV(self.prms.output_dir,
                              f'gridded_counts.txt',
                              counts,
                              col_titles)
        self.write_matrix_CSV(self.prms.output_dir,
                              f'gridded_densities.txt',
                              cell_densities_km2,
                              col_titles)
        return counts, cell_densities_km2


if __name__ == "__main__":

    if len(sys.argv[1:]) >= 1:
        configfile = sys.argv[1]
    else:
        configfile = 'parameters.txt'

    start_time = time.time()
    algo = VoronoiSmoothingAlgorithm(configfile)
    counts, densities = algo.run()
    print(f'\n>> Execution time: {time.time() - start_time:.3f} s.')
