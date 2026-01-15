import sys
import os
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from argparse import ArgumentParser

from lib.ioutils import (ParameterSet, 
                         load_bins, 
                         change_zvalue_in_polygon_file,
                         load_grid,
                         load_fmd_file,
                         load_polygons)
from lib.grt_ml import Dutfoy2020_Estimator
from lib.geoutils import convert_to_EPSG

COORD_PRECISION = 1E-6  # Precision used for the comparison of cell coordinates

class TruncatedGRestimator():
    
    def __init__(self):
        self.file_densities = None
        self.file_bins = None
        self.file_fmd = None
        self.file_prior_b = None
        self.file_csv = None    
        self.file_GMT_a = None
        self.file_GMT_b = None
        self.file_GMT_da = None
        self.file_GMT_db = None
        self.file_GMT_rho = None
        self.file_GMT_mmax = None
        self.file_GMT_mc = None
        self.densities = None
        self.areas = None
        self.ncells = None
        self.nbins = None
        self.bins = None
        self.cellinfo = None
        self.bin_durations = None
        self.prior_b = None
        self.target_area_for_a = None
        self.grt_params = None
        self.ibins = None
        self.mmin = None
        self.mmax = None

    def load_densities(self, filename, scaling_factor=1.0, rescale_to_polygons_areas=None):
        """
        Load densities from input file and eventually apply scaling to density values

        :param filename, str: path to the density file
        :poram scaling_factor, float: constant multiplicative scaling value applied to loaded densities values
        :param rescale_to_polygons_areas, None or numpy.ndarray, if not None, cell-wise multiplicative density scaling factor
        """
        self.file_densities = filename
        # We load densities rescaled to 1 km^2 area:
        self.densities, self.ncells, self.nbins, self.bin_ids = load_grid(
            self.file_densities,
            scaling_factor=scaling_factor)
        nl, nc = self.densities.shape
        # If specified, we scale densities to polygons areas (in km^2):
        if rescale_to_polygons_areas is None:
            areas = np.ones((nl, nc - 2))
        else:
            areas = np.tile(np.reshape(rescale_to_polygons_areas, (-1, 1)), (1, nc - 2))
        self.densities[:, 2:] *= areas

    def load_bins(self, filename):
        self.file_bins = filename
        magbins = load_bins(filename)
        self.ibins = [k for k, bin_id in enumerate(magbins[:, 0]) if int(bin_id) in self.bin_ids]
        self.bins = dict()
        self.bins.update({'id': magbins[self.ibins, 0].astype(int)})  # Used in the naming of output files
        self.bins.update({'mins': magbins[self.ibins, 1]})
        self.bins.update({'maxs': magbins[self.ibins, 2]})
        self.bins.update({'mids': self.bins['mins'] + 0.5 * (self.bins['maxs'] - self.bins['mins'])})
        self.bins.update({'durations': magbins[self.ibins, 4] - magbins[self.ibins, 3]})
        if magbins.shape[0] != len(self.ibins):
            print(f'{filename}:: {magbins.shape[0] - len(self.ibins)} bins missing in density file. ' +
                  f'Will only use {len(self.ibins)} bins.')

    def load_fmd_info(self, filename=None, verbose=True):
        """
        Load FMD information (Mmin, Mmax, and optionally bin durations)
        Format: LON; LAT; MMIN; MMAX [; BIN_1_DURATION; ...; BIN_n_DURATION]
        Note:
            Missing MMax can be indicated as -9 or NaN in the input file.
        """
        self.file_fmd = filename
        lons = self.densities[:, 0]
        lats = self.densities[:, 1]
        self.cellinfo, self.bin_durations = load_fmd_file(
            self.file_bins,
            lons,
            lats,
            fmd_file=self.file_fmd,
            ibins=self.ibins,
            mmin=self.mmin,
            mmax=self.mmax,
            coord_precision=COORD_PRECISION,
            verbose=verbose)

    def load_prior_on_b(self, filename=None, b_mean=1.0, b_std=1.0, verbose=True):
        """
        Load a-priori information on b-value. For unspecified cells, default prior is b=1, std_b=1.
        Note:
            When a prior file is provided, if both b_mean and b_std are equal to -9 or 0, this is interpreted
            as an absence of prior for the corresponding cell.
        """
        # NB: PRIORINFO Format:
        # priorinfo[i,:] = [lon, lat, b_mean, b_std]
        prior = np.ones((self.ncells, 2))  # Column order: [mean_b, std_dev_b]
        if filename:
            self.file_prior_b = filename
            print(f'>> Prior information on b-values read from "{filename}"')
            priorinfo = np.loadtxt(filename, delimiter=';', comments='#')
            for i in range(self.ncells):
                j = np.where((np.abs(priorinfo[:, 0] - self.densities[i, 0]) < COORD_PRECISION) & \
                             (np.abs(priorinfo[:, 1] - self.densities[i, 1]) < COORD_PRECISION) )[0]
                if len(j) == 0:
                    if verbose:
                        print(f'{filename}:: No match for cell with centroid at ' +
                              f'({self.densities[i, 0]:.6f}; {self.densities[i, 1]}) --> Set prior b = 1.0 +/- 1.0')
                elif len(j) > 1:
                    raise ValueError(f'{filename}:: Found several lines ({j}) matching coordinates'
                                     + f'({self.densities[i, 0]:.6f}; {self.densities[i, 1]}) within '
                                     + f'given precision ({COORD_PRECISION})')
                else:
                    prior[i, :] = priorinfo[j, 2:]  # [mean_b, std_dev_b]
        else:
            self.file_prior_b = 'command-line'
            print(f'>> Homogeneous prior on b-values over the domain: <b> = {b_mean:.2f}, std = {b_std:.2f}')
            prior[:, 0] = b_mean
            prior[:, 1] = b_std
        self.prior_b = prior

    def _select_cells(self, condition):
        n_init = len(condition)
        assert n_init == self.ncells
        bin_durations = self.bin_durations[condition, :]
        cellinfo = self.cellinfo[condition, :]
        densities = self.densities[condition, :]
        prior_b = self.prior_b[condition, :]
        i_kept = np.nonzero(condition)[0]
        i_removed = np.nonzero(np.logical_not(condition))[0]
        ncells = len(i_kept)
        print(f'>> Kept {ncells} cells from the original grid of {n_init} cells')
        return bin_durations, cellinfo, densities, prior_b, i_kept, i_removed

    def _ML_estimation(self, skip_missing_priors=False, auto_mc=False, b_truncation=None, scaling_area_km2=None):
        """
        Maximum-likelihood estimation of (a, b) parameters (Dutfoy, 2021; Weichert, 1980)

        :return:
        """
        for i in tqdm(range(self.ncells)):
            cell_intensities = np.reshape(self.densities[i, 2:], (self.nbins,))
            cell_durations = np.reshape(self.bin_durations[i, :], (self.nbins,))
            dm = self.bins['maxs'][0] - self.bins['mins'][0]  # NB: Constant magnitude bin width is assumed
            lon = self.cellinfo[i, 0]
            lat = self.cellinfo[i, 1]
            mmin = self.cellinfo[i, 2]
            mmax = self.cellinfo[i, 3]

            # Eventually, remove unused bins (with durations <= 0):
            ib = (self.bin_durations[i, :] > 0) & (self.bins['mins'] >= mmin)
            # Defensive-programming:
            # Check that all bins have a definite and positive duration:
            assert np.any(ib)
            cell_intensities = cell_intensities[ib]
            cell_durations = cell_durations[ib]
            cell_mmid = self.bins['mids'][ib]



            # Eventually, define automatically the completeness threshold:
            if auto_mc:
                imc = cell_intensities.argmax()
                mc = cell_mmid[imc] - dm / 2
            else:
                imc = np.where(cell_mmid - dm / 2 >= mmin)[0][0]
                mc = mmin

            ll = Dutfoy2020_Estimator(cell_mmid[imc:],
                                      cell_durations[imc:],
                                      cell_intensities[imc:],
                                      mc,
                                      mmax,
                                      dm)

            def ab_estimation_method(*input_args, bounds_b=None):
                if (len(input_args) == 0) and (bounds_b is None):
                    return ll.find_optimal_ab_no_prior()
                elif (len(input_args) > 0) and (bounds_b is None):
                    return ll.find_optimal_ab_with_normal_prior(*input_args)
                elif (len(input_args) == 0) and (bounds_b is not None):
                    return ll.find_optimal_ab_no_prior_b_truncated(bounds_b)
                elif (len(input_args) > 0) and (bounds_b is not None):
                    return ll.find_optimal_ab_with_truncated_normal_prior(*input_args, bounds_b=bounds_b)

            if isinstance(self.file_prior_b, str):
                bmean = self.prior_b[i, 0]
                bstd = self.prior_b[i, 1]
                if ((bmean == -9.0) and (bstd == -9.0)) or ((bmean == 0.0) and (bstd == 0.0)):
                    # Identified a flag indicating the absence of prior for this cell:
                    if skip_missing_priors:
                        self.grt_params[i, :] = np.array([lon, lat] + [np.nan] * 7)
                        continue
                    else:
                        #a, b, rho, cov = ll.find_optimal_ab_no_prior()
                        a, b, rho, cov = ab_estimation_method(bounds_b=b_truncation)
                else:
                    #a, b, rho, cov = ll.find_optimal_ab_with_normal_prior(bmean, bstd)
                    a, b, rho, cov = ab_estimation_method(bmean, bstd, bounds_b=b_truncation)
            else:
                #a, b, rho, cov = ll.find_optimal_ab_no_prior()
                a, b, rho, cov = ab_estimation_method(bounds_b=b_truncation)

            stdb = np.sqrt(cov[0, 0])
            stda = np.sqrt(cov[1, 1])

            # Eventually, scale a-value to target area (do NOT scale stda, stdb and b!):
            if scaling_area_km2 is None:
                target_area = self.areas[i]
            else:
                a = np.log10(scaling_area_km2 / self.areas[i] * 10 ** (a))
                target_area = scaling_area_km2
            self.grt_params[i, :] = np.array([lon, lat, a, b, stda, stdb, rho, mc, target_area])


    def run(self, options, print_warnings=False, b_truncation=None, target_area_km2=None):
        """
        Evaluate a and b parameters of the truncated GR relationship in each cell
        """
        self.grt_params = np.zeros((self.ncells, 9))
        if print_warnings:
            print(f'Warning !! Constant width is assumed for FMD bins in each cell (but can vary cell-wise)')

        self._ML_estimation(skip_missing_priors=options['skip_missing_priors'],
                            auto_mc=options['auto_mc'],
                            b_truncation=b_truncation,
                            scaling_area_km2=target_area_km2)

    def write_to_csv(self, filename):
        """
        Save results in CSV format
        """
        self.file_csv = filename
        np.savetxt(filename,
                   self.grt_params, 
                   header='; '.join(['lon', 'lat', 'a', 'b', 'da', 'db', 'rho_ab', 'mc', 'area_in_km2']),
                   delimiter='; ')
        print(f'{filename}:: saved Gutenberg-Richter parameters for {self.ncells} cells')
    
    def write_to_GMT_ASCII_tables(self, directory='.'):
        """
        Save results into GMT ASCII polygon files
        """
        template_cell_file = os.path.join(directory, 
                                          f'density_bin_{self.bins["id"][0]}.txt')
        self.file_GMT_a = os.path.join(directory, 'a_cells.txt')
        change_zvalue_in_polygon_file(template_cell_file, self.file_GMT_a, self.grt_params[:, 2])
        self.file_GMT_b = os.path.join(directory, 'b_cells.txt')
        change_zvalue_in_polygon_file(template_cell_file, self.file_GMT_b, self.grt_params[:, 3])
        self.file_GMT_da = os.path.join(directory, 'da_cells.txt')
        change_zvalue_in_polygon_file(template_cell_file, self.file_GMT_da, self.grt_params[:, 4])
        self.file_GMT_db = os.path.join(directory, 'db_cells.txt')
        change_zvalue_in_polygon_file(template_cell_file, self.file_GMT_db, self.grt_params[:, 5])
        self.file_GMT_rho = os.path.join(directory, 'rho_ab_cells.txt')
        change_zvalue_in_polygon_file(template_cell_file, self.file_GMT_rho, self.grt_params[:, 6])
        self.file_GMT_mc = os.path.join(directory, 'mc_cells.txt')
        change_zvalue_in_polygon_file(template_cell_file, self.file_GMT_mc, self.grt_params[:, 7])
        self.file_GMT_mmax = os.path.join(directory, 'mmax_cells.txt')
        change_zvalue_in_polygon_file(template_cell_file, self.file_GMT_mmax, self.cellinfo[:, 3])


if __name__ == "__main__":

    # Read input arguments:
    parser = ArgumentParser(
        description="Compute (a, b) parameters of the frequency-magnitude distribution in each cell " \
                    + "(NB: These parameters are obtained from densities rescaled at the area of each " \
                    + "individual cell, before an eventual rescaling to the target reference area, unless " \
                    + "option -s is set, in order to preserve the coherency of covariance estimates).")
    parser.add_argument("configfile",
                        nargs='?',
                        default="parameters.txt",
                        help="Configuration file")

    parser.add_argument("-b", "--b-prior",
                        help="Specify homogeneous prior on b (mean, std. dev.) over the spatial domain.",
                        nargs=2,
                        type=float)

    parser.add_argument("-s", "--rescale-to-cell-area",
                        help='If set, rescale densities (and a-values) to each cell/polygon area. ' \
                             + 'Otherwise, keep parameters scaled for the area specified (in km^2) ' \
                             + 'in configuration file (see parameter "density_scaling_factor").',
                        action='store_true')

    parser.add_argument("--b-truncation",
                        help="Set lower and upper truncation for b-values",
                        nargs=2,
                        type=float)

    parser.add_argument("--mmin",
                        help="Set the minimum magnitude of the (un-)truncated GR model for all pixels (overwrites FMD file, if provided)",
                        default=None,
                        type=float)

    parser.add_argument("--mmax",
                        help="Set the maximum magnitude of the truncated GR model for all pixels (overwrites FMD file, if provided)",
                        default=None,
                        type=float)

    args = parser.parse_args()
    
    # Load parameters:
    prms = ParameterSet()
    prms.load_settings(args.configfile)

    # Load data:
    inputfile = os.path.join(prms.output_dir, 'gridded_densities.txt')
    estim = TruncatedGRestimator()
    estim.mmin = args.mmin
    estim.mmax = args.mmax
    pols, _ = load_polygons(os.path.join(prms.output_dir, 'counts_bin_1.txt'))
    pols_m = convert_to_EPSG(pols, in_epsg=prms.input_epsg, out_epsg=prms.internal_epsg)
    polareas = np.array([pol.area * (prms.epsg_scaling2km ** 2) for pol in pols_m.geoms])  # in km^2
    area_scaling = 1 / prms.density_scaling_factor
    estim.areas = polareas

    # NB: Calling next function affects a value to estim.ncells
    estim.load_densities(inputfile,
                         scaling_factor=area_scaling,
                         rescale_to_polygons_areas=polareas)
    estim.load_bins(prms.bins_file)

    # Load FMD information (Mmin, Mmax, and optionally bin durations):
    estim.load_fmd_info(filename=prms.fmd_info_file, verbose=prms.is_verbose)

    # Load prior information on b-value:
    if (args.b_prior is not None) and (prms.prior_b_info_file is not None):
        raise ValueError('Error: Cannot use option "-b" when "file_for_prior_b_information" ' +
                         'is already specified in configuration file')
    elif prms.prior_b_info_file is not None:
        estim.load_prior_on_b(filename=prms.prior_b_info_file, verbose=prms.is_verbose)
    elif isinstance(args.b_prior, list):
        estim.load_prior_on_b(b_mean=args.b_prior[0], b_std=args.b_prior[1], verbose=prms.is_verbose)
    else:
        # No prior on b-values:
        print('>> No prior for b-values over the domain')
        estim.file_prior_b = None

    # Set options:
    opts = {'skip_missing_priors': prms.skip_ab_if_missing_priors,
           'auto_mc': prms.is_mc_automatic}

    # Estimate G-R parameters over cells:
    if args.rescale_to_cell_area:
        estim.run(opts,
                  print_warnings=False,
                  b_truncation=args.b_truncation,
                  target_area_km2=None)
    else:
        estim.run(opts,
                  print_warnings=False,
                  b_truncation=args.b_truncation,
                  target_area_km2=prms.density_scaling_factor)

    # Save results:
    estim.write_to_csv(os.path.join(prms.output_dir, 'ab_values.txt'))
    estim.write_to_GMT_ASCII_tables(directory=prms.output_dir)
