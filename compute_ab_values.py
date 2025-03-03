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
                         load_fmd_file)
from lib.grt_ml import Dutfoy2020_Estimator
from lib.grt_pl import penalized_likelihood, load_mcmc_results

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

    def load_densities(self, filename, scaling_factor=1.0):
        self.file_densities = filename
        self.densities, self.ncells, self.nbins, self.bin_ids = load_grid(
            self.file_densities,
            scaling_factor=scaling_factor)

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

    def load_fmd_info(self, filename=None):
        """
        Load FMD information (Mmin, Mmax, and optionally bin durations)
        Format: LON; LAT; MMIN; MMAX [; BIN_1_DURATION; ...; BIN_n_DURATION]
        Note:
            Missing MMax can be indicated as -9 or NaN in the input file.
        """

        """
        # Initialize bin durations:
        self.bin_durations = np.ones((self.ncells, self.nbins))
        for i in range(self.ncells):
            # NB: bin list should already be decimated to those included in density file
            self.bin_durations[i, :] = self.bins['durations']

        cellinfo = np.ones((self.ncells, 4))  # For series of (lon, lat, mmin, mmax)
        if isinstance(filename, str):
            print(f'>> Read limits (and optionally, durations) of freq.-mag. '
                  + f'distributions in "{filename}"')
            gridinfo = np.loadtxt(filename, delimiter=';')
            # Check GRIDINFO Format:
            # gridinfo[i,:] = [lon, lat, mmin, mmax, dur_i, dur_i+1, ..., dur_N]
            if gridinfo.shape[1] < 3:
                raise ValueError(f"{filename}:: Missing columns (at least 3 required: lon, lat, mmin)")
            elif gridinfo.shape[1] == 3:
                print(f'{filename}:: No max. magnitude specified, will use Mmax = Inf (untruncated model)')
                print(f'{filename}:: No durations. Will use bin durations from {self.file_bins}')
            elif gridinfo.shape[1] == 4:
                print(f'{filename}:: Will use max. magnitudes as specified in file')
                print(f'{filename}:: No durations. Will use bin durations from {self.file_bins}')
            elif gridinfo.shape[1] > 4:
                print(f'{filename}:: Specified bin durations will supercede those given in "{self.file_bins}"')

            for i in range(self.ncells):
                j = np.where((np.abs(gridinfo[:, 0] - self.densities[i, 0]) < COORD_PRECISION) & \
                             (np.abs(gridinfo[:, 1] - self.densities[i, 1]) < COORD_PRECISION) )[0]
                if len(j) == 0:
                    print(f'{filename}:: Cannot find cell with coordinates ' +
                          f'({self.densities[i, 0]:.6f}; {self.densities[i, 1]})')
                    continue
                elif len(j) > 1:
                    raise ValueError(f'{filename}:: Found several lines with coordinates matching '
                                     + f'({self.densities[i, 0]:.6f}; {self.densities[i, 1]}): {j}')

                cellinfo[i, :3] = gridinfo[j, :3]  # Copy (lon, lat, Mmin) values
                if gridinfo.shape[1] > 3:
                    # Manage Mmax values (missing or not):
                    if (gridinfo[j, 3].item() == -9.0) or np.isnan(gridinfo[j, 3]):
                        cellinfo[i, 3] = np.inf  # untruncated G-R model
                    else:
                        cellinfo[i, 3] = gridinfo[j, 3].item()  # Copy Mmax
                else:
                    cellinfo[i, 3] = np.inf  # untruncated G-R model

                # Update bin durations, if available in gridinfo:
                if gridinfo.shape[1] > 4:
                    self.bin_durations[i, :] = gridinfo[j, [4 + k for k in self.ibins]]
        else:
            print('>> Missing option "file_for_FMD_limits_and_durations" in configuration file')
            print('>> MMAX: No upper truncation (i.e., MMAX = inf.)')
            if self.mmin is None:
                self.mmin = min(self.bins['mins'])
                print(f'>> MMIN: Use the minimum lower bound of intervals read in "{self.file_bins}": {self.mmin:.2f}')
            else:
                print(f'>> MMIN: Use the minimum magnitude given in command-line: {self.mmin:.2f}')
            cellinfo[:, :2] = self.densities[:, :2]
            cellinfo[:, 2] = self.mmin
            cellinfo[:, 3] = np.inf
        self.cellinfo = cellinfo
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
            coord_precision=COORD_PRECISION)

    def load_prior_on_b(self, filename=None, b_mean=1.0, b_std=1.0):
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

    def _ML_estimation(self, skip_missing_priors=False, auto_mc=False, b_truncation=None):
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
            ib = (self.bin_durations[i, :] > 0)
            cell_intensities = cell_intensities[ib]
            cell_durations = cell_durations[ib]
            cell_mmid = self.bins['mids'][ib]

            # Eventually, define automatically the completeness threshold:
            if auto_mc:
                imc = cell_intensities.argmax()
                mc = self.bins['mids'][imc] - dm / 2
            else:
                imc = np.where(self.bins['mids'] - dm / 2 >= mmin)[0][0]
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
                        self.grt_params[i, :] = np.array([lon, lat] + [np.nan] * 6)
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
            self.grt_params[i, :] = np.array([lon, lat, a[0, 0], b[0, 0], stda, stdb, rho, mc])

    def _PL_estimation(self, max_mcmc_trials=1E6, global_b_prior_mean=1.0, global_b_prior_std=0.3,
                       apply_smoothing=True, b_truncation=None):
        """
        Penalized-likelihood estimation of (a, b) parameters (EPRI, 2012).
        This algorithm inverts jointly paramters for all cells of the domain,
        using a MCMC algorithm. Thus, convergence can be long...
        """
        # First, remove cells with null density in the first bin
        # (indicating null density in every bins):
        condition = (self.densities[:, 2] != 0.0) & (self.prior_b[:, 0] != -9.0)
        bin_durations, cellinfo, densities, prior_b , i_kept, i_removed = \
            self._select_cells(condition)
        m = cellinfo.shape[0]  # Number of cells with density > 0

        areas = np.ones((m, ))
        imc = np.where(self.bins['mins'] >= self.mmin)[0][0]
        bins = np.array([self.bins['mins'], self.bins['maxs']]).T
        nb = bins.shape[0]
        bins_w = np.ones((nb,))  # Can be changed to favor lower-/higher-magntiude bins
        m0 = self.mmin

        # Build a PMF composed of a single MMAX value:
        mmax_pmf = np.ones((m, 1, 2))
        for k in range(m):
            mmax_pmf[k, 0, :] = np.array([cellinfo[k, 3], 1.0])

        # Define prior on parameters:
        prior = dict()
        prior['b_mean'] = global_b_prior_mean
        prior['b_std'] = global_b_prior_std
        lograte_m0 = np.log10(densities[:, 2])
        beta_prior = prior_b[:, 0] * np.log(10)
        beta_std_prior = prior_b[:, 1] * np.log(10)
        prior['mins'] = np.array((lograte_m0 - 2.0).tolist() + (beta_prior - 2 * beta_std_prior).tolist() + [0.1, 0.1])
        prior['maxs'] = np.array((lograte_m0 + 2.0).tolist() + (beta_prior + 2 * beta_std_prior).tolist() + [1, 4])
        if b_truncation is not None:
            # Apply truncation to beta values if needed:
            prior['mins'][m:(2 *m)] = np.fmin(prior['mins'][m:(2 * m)], b_truncation[0] * np.log(10) * np.ones((m,)))
            prior['maxs'][m:(2 * m)] = np.fmax(prior['maxs'][m:(2 * m)], b_truncation[1] * np.log(10) * np.ones((m,)))

        n_warmup = round(max_mcmc_trials * 0.01)
        penalized_likelihood(densities[:, (2 + imc):],
                             areas,
                             cellinfo[:, 0],
                             cellinfo[:, 1],
                             bins[imc:, :],
                             bin_durations[:, imc:],
                             mmax_pmf,
                             bins_w[imc],
                             m0=m0,
                             max_trials=max_mcmc_trials,
                             n_init=n_warmup,
                             collection_interval='auto',
                             prior=prior,
                             proposal={'fun_type': 'uniform-local', 'sigma': 0.1, 'strategy': 'incremental'},
                             output_file='mcmc_samples.out',
                             only_sample_prior=True,
                             seed=None,
                             apply_smoothing=apply_smoothing)
        lons, lats, index, post, accept, a, b, sm_beta, sm_lograte = \
            load_mcmc_results(file="mcmc_samples.out",
                              subsampling_step=1,
                              grid_reshape=None,
                              additive_scaling_for_a=0.0)

        # Compute summary statistics:
        sm_beta_mean = np.mean(sm_beta)
        sm_beta_std = np.std(sm_beta)
        sm_lograte_mean = np.mean(sm_lograte)
        sm_lograte_std = np.std(sm_lograte)
        print('# Optimal smoothing control parameters:')
        print(f'  --> for beta: {sm_beta_mean} +/- {sm_beta_std}')
        print(f'  --> for log10(rate): {sm_lograte_mean} +/- {sm_lograte_std}')

        a_mean = np.mean(a, axis=0)
        a_std = np.mean(a, axis=0)
        b_mean = np.mean(b, axis=0)
        b_std = np.mean(b, axis=0)
        rho = np.array([np.corrcoef(a[:, i], b[:, i], rowvar=True)[0, 1] for i in range(m)])
        for j in range(m):
            self.grt_params[i_kept[j], :] = np.array([lons[j],
                                                      lats[j],
                                                      a_mean[j],
                                                      b_mean[j],
                                                      a_std[j],
                                                      b_std[j],
                                                      rho[j],
                                                      m0])
        for j in range(len(i_removed)):
            self.grt_params[i_removed[j], :] = np.array([self.densities[i_removed[j], 0],
                                                         self.densities[i_removed[j], 1],
                                                         np.nan,
                                                         np.nan,
                                                         np.nan,
                                                         np.nan,
                                                         np.nan,
                                                         np.nan])

    def run(self, method, options, print_warnings=False, b_truncation=None):
        """
        Evaluate a and b parameters of the truncated GR relationship in each cell
        """
        self.grt_params = np.zeros((self.ncells, 8))
        if print_warnings:
            print(f'Warning !! Constant width is assumed for FMD bins in each cell (but can vary cell-wise)')

        if method == 'ML':
            self._ML_estimation(skip_missing_priors=options['skip_missing_priors'],
                                auto_mc=options['auto_mc'],
                                b_truncation=b_truncation)
        elif method == 'PL':
            self._PL_estimation(max_mcmc_trials=options['max_trials'],
                                global_b_prior_mean=options['global_b_mean'],
                                global_b_prior_std=options['global_b_std'],
                                b_truncation=b_truncation)
        else:
            raise ValueError(f'Unrecognized method: "{method}"')

    
    def write_to_csv(self, filename):
        """
        Save results in CSV format
        """
        self.file_csv = filename
        np.savetxt(filename,
                   self.grt_params, 
                   header='; '.join(['lon', 'lat', 'a', 'b', 'da', 'db', 'rho_ab', 'mc']),
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
        description="Compute (a, b) parameters of the frequency-magnitude distribution in each cell")
    parser.add_argument("configfile",
                        nargs='?',
                        default="parameters.txt",
                        help="Configuration file")

    parser.add_argument("-b", "--b-prior",
                        help="Specify homogeneous prior on b (mean, std. dev.) over the spatial domain. " +
                             "Supercedes global priors set using option '--pl-parameters' if method PL is used (-m)",
                        nargs=2,
                        type=float)

    parser.add_argument("--b-truncation",
                        help="Set lower and upper truncation for b-values",
                        nargs=2,
                        type=float)

    parser.add_argument("-m", "--method",
                        help="Method for estimation of MFD parameters (a, b): "+
                             "'ML' for maximum-likelihood (Weichert, 1980), "+
                             "'PL' for penalized-likelihood (EPRI, 2012)",
                        choices=['ML', 'PL'],
                        default='ML',
                        type=str)

    parser.add_argument('--par',
                        help='Parameters for the penalized-likelihood approach: N_trials, Global priors on b (mean, std. dev.)',
                        default=[1E6, 1.0, 0.3],
                        nargs=3,
                        type=float)

    parser.add_argument("--mmin",
                        help="Set the minimum magnitude used to estimate FMD parameters",
                        default=None,
                        type=float)
    # TODO: Remove priors from --par option --> Keep only --b-prior option and make b-prior option mandatory when --par is used
    args = parser.parse_args()
    
    # Load parameters:
    prms = ParameterSet()
    prms.load_settings(args.configfile)

    # Load data:
    inputfile = os.path.join(prms.output_dir, 'gridded_densities.txt')
    estim = TruncatedGRestimator()
    estim.mmin = args.mmin
    estim.load_densities(inputfile, scaling_factor=1.0)  # NB: Scaling already applied in voronoi2density.py
    estim.load_bins(prms.bins_file)
    
    # Load FMD information (Mmin, Mmax, and optionally bin durations):
    estim.load_fmd_info(prms.fmd_info_file)

    # Load prior information on b-value:
    if (args.b_prior is not None) and (prms.prior_b_info_file is not None):
        raise ValueError('Error: Cannot use option "-b" when "file_for_prior_b_information" ' +
                         'is already specified in configuration file')
    elif prms.prior_b_info_file is not None:
        estim.load_prior_on_b(filename=prms.prior_b_info_file)
    elif isinstance(args.b_prior, list):
        estim.load_prior_on_b(b_mean=args.b_prior[0], b_std=args.b_prior[1])
    else:
        # No prior on b-values:
        print('>> No prior for b-values over the domain')
        estim.file_prior_b = None

    if (args.b_prior is not None) and (args.method == 'PL'):
        # In this case, option "b_prior" supercedes values in "pl-parameters":
        args.par[1] = args.b_prior[0]
        args.par[2] = args.b_prior[1]

    # Set options for the chosen method:
    if args.method == 'PL':
        opts = {'max_trials': args.par[0],
                'global_b_mean': args.par[1],
                'global_b_std': args.par[2]
                }
    elif args.method == 'ML':
        opts = {'skip_missing_priors': prms.skip_ab_if_missing_priors,
                'auto_mc': prms.is_mc_automatic}

    # Estimate G-R parameters over cells:
    estim.run(args.method,
              opts,
              print_warnings=False,
              b_truncation=args.b_truncation)
    
    # Save results:
    estim.write_to_csv(os.path.join(prms.output_dir, 'ab_values.txt'))
    estim.write_to_GMT_ASCII_tables(directory=prms.output_dir)
