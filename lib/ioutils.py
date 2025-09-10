"""
I/O MANAGEMENT ROUTINES FOR ZONELESS_VORONOI.PY
"""
import os
import numpy as np
from shapely import GeometryCollection, Point, MultiPoint, prepare, Polygon, MultiPolygon


class ParameterSet():

    def __init__(self) -> None:
        self.epicenters_file = None
        self.bounds_file = None
        self.bins_file = None
        self.figures_dir = 'figures'
        self.output_dir = '.'
        self.input_epsg = None
        self.internal_epsg = None
        self.epsg_scaling2km = None
        self.mesh_type = None  # 'regular' or 'polygons'
        self.mesh_step = None
        self.mesh_step_unit = None
        self.mesh_file = None
        self.is_verbose = False
        self.fmd_info_file = None
        self.prior_b_info_file = None
        self.density_scaling_factor = 1.0
        self.is_mc_automatic = False
        self.nb_bootstrap_samples = 0  # When 0: Bootstrap resampling deactivated
        self.perturb_magnitudes = False  # Activate/De-activate the random perturbation of magnitudes
        self.b_value_for_correction_term = 1.0  # b-value used to correct the bias of perturbed magnitudes
        self.save_realizations = False
        self.nb_parallel_tasks = None
        self.subdivide_polygons = False  # Activate/De-activate Voronoi polygon sub-division into triangles using the germ as a common vertex
    
    def load_settings(self, filename, mandatory_fields=[]) -> None:
        """
        Load settings from ASCII file
        """
        print(f'>> Loading configuration file: {filename}...')
        if not os.path.exists(filename):
            raise ValueError(f'Error: file "{filename}" not found')
        with open(filename, 'rt') as fp:
            lines = fp.readlines()
            for line in lines:
                items = line.strip().split(' ')  # Remove trailing "\n" and split w.r. to space character
                if items[0] == 'file_for_epicenters:':
                    self.epicenters_file = items[1]
                    print(f'{filename}:: Load epicenters from file "{self.epicenters_file}"')

                elif items[0] == 'file_for_geographical_bounds:':
                    self.bounds_file = items[1]
                    print(f'{filename}:: Load geographical bounds from file "{self.bounds_file}"')

                elif items[0] == 'file_for_polygonal_cells_definition:':
                    if self.mesh_type == 'regular':
                        raise ValueError('Incoherent polygonal-mesh file parameter with pre-defined regular-grid discretization parameter. ' +
                                         'Remove optional grid discretization parameter.')
                    self.mesh_file = items[1]
                    self.mesh_type = 'polygons'
                    print(f'{filename}:: Load polygonal cells from file "{self.mesh_file}"')

                elif items[0] == 'file_for_magnitude_bins:':
                    self.bins_file = items[1]
                    print(f'{filename}:: Load magnitude bins from file "{self.bins_file}"')
                
                elif items[0] == 'file_for_FMD_limits_and_durations:':
                    self.fmd_info_file = items[1]
                    print(f'{filename}:: Load FMD limits and durations from file "{self.fmd_info_file}"')

                elif items[0] == 'file_for_prior_b_information:':
                    self.prior_b_info_file = items[1]
                    print(f'{filename}:: Load a-priori b-value information from file "{self.prior_b_info_file}"')

                elif items[0] == 'output_directory_for_files:':
                    self.output_dir = items[1]
                    if not os.path.isdir(self.output_dir):
                        os.mkdir(self.output_dir)
                    print(f'{filename}:: Output directory for files is "{self.output_dir}"')
                
                elif items[0] == 'output_directory_for_figures:':
                    self.figures_dir = items[1]
                    if not os.path.isdir(self.figures_dir):
                        os.mkdir(self.figures_dir)
                    print(f'{filename}:: Output directory for figures is "{self.figures_dir}"')

                elif items[0] == 'input_CRS:':
                    self.input_epsg = items[1]
                    print(f'{filename}:: CRS for input coordinates: {self.input_epsg}')
                
                elif items[0] == 'internal_equal_area_CRS:':
                    self.internal_epsg = items[1]
                    print(f'{filename}:: CRS for internal area computations: {self.internal_epsg}')
                
                elif items[0] == 'unit_for_internal_CRS_coordinates:':
                    if items[1] in ['m', 'meters', 'meter']:
                        self.epsg_scaling2km = 0.001
                    elif items[1] in ['km', 'kilometers', 'kilometer']:
                        self.epsg_scaling2km = 1.0
                    elif items[1] in ['feet', 'ft']:
                        self.epsg_scaling2km = 0.0003048
                    print(f'{filename}:: Scaling coef. to convert internal coordinates in km: {self.epsg_scaling2km}')
        
                elif items[0] == 'mesh_discretization_step:':
                    if self.mesh_type == 'polygons':
                        raise ValueError('Incoherent regular-grid discretization with pre-defined polygonal-mesh type. ' +
                                         'Remove optional polygonal mesh file parameter.')
                    self.mesh_step = float(items[1])
                    self.mesh_step_unit = items[2]  # km or deg only!
                    self.mesh_type = 'regular'
                    mandatory_fields.append('bounds_file')
                    print(f'{filename}:: Zoneless grid discretization step: {self.mesh_step} {self.mesh_step_unit}')

                elif items[0] == 'density_scaling_factor:':
                    self.density_scaling_factor = float(items[1])
                    print(f'{filename}:: Density scaling factor set to {self.density_scaling_factor}')
                
                elif items[0] == 'skip_ab_if_missing_priors:':
                    self.skip_ab_if_missing_priors = (items[1].strip().lower() == "true")
                    if self.skip_ab_if_missing_priors: 
                        print(f'{filename}:: Skip (a,b) evaluation if missing priors')
                    else:
                        print(f'{filename}:: Do NOT skip (a,b) evaluation if missing priors')

                elif items[0] == 'enable_verbosity:':
                    self.is_verbose = (items[1].strip().lower() == 'true')
                    if self.is_verbose:
                        print(f'{filename}:: Enabled verbosity')

                elif items[0] == 'define_completeness_automatically:':
                    self.is_mc_automatic = (items[1].strip().lower() == 'true')
                    if self.is_mc_automatic:
                        print(f'{filename}:: Completeness threshold determined automatically (above Mmin)')

                elif items[0] == 'nb_bootstrap_samples:':
                    self.nb_bootstrap_samples = int(items[1])
                    if self.nb_bootstrap_samples > 0 :
                        print(f'{filename}:: Apply bootstrap resampling with {self.nb_bootstrap_samples} samples')

                elif items[0] == 'perturb_magnitudes:':
                    self.perturb_magnitudes = (items[1].strip().lower() == 'true')
                    if self.perturb_magnitudes :
                        print(f'{filename}:: Activate the random perturbation of magnitudes in the bootstrapping process')
                    elif self.nb_bootstrap_samples > 0:
                        print(f'{filename}:: De-activate the random perturbation of magnitudes in the bootstrapping process')

                elif items[0] == 'b_value_to_remove_bias_on_perturbed_magnitudes:':
                    self.b_value_for_correction_term = float(items[1])
                    print(f'{filename}:: Specific b-value to correct the bias on perturbed magnitudes: {self.b_value_for_correction_term}')

                elif items[0] == 'save_bootstrap_realizations:':
                    self.save_realizations = (items[1].strip().lower() == 'true')
                    if self.save_realizations :
                        print(f'{filename}:: Save results associated with every boostrap sample')

                elif items[0] == 'nb_parallel_tasks:':
                    self.nb_parallel_tasks = int(items[1])
                    if self.nb_parallel_tasks > 0 :
                        print(f'{filename}:: Use {self.nb_parallel_tasks} parallel tasks')
                    else:
                        self.nb_parallel_tasks = None  # Defensive programming

                elif items[0] == 'subdivide_polygons:':
                    self.subdivide_polygons = (items[1].strip().lower() == 'true')
                    if self.subdivide_polygons :
                        print(f'{filename}:: Subdivides Voronoi polygons in triangles')
                    else:
                        print(f'{filename}:: De-activated Voronoi polygon subdivision in triangles')

        # Defensive programming: check for missing specifications
        for attname in ['epicenters_file',
                        'bins_file',
                        'input_epsg', 
                        'internal_epsg', 
                        'epsg_scaling2km',
                        'mesh_type'] + mandatory_fields:
            if getattr(self, attname) is None:
                raise_str = f'{filename}:: Missing specification for variable "{attname}". ' \
                    + 'See function lib.ioutils.load_settings().'
                raise ValueError(f'{raise_str}')

        # Special case for the bootstrapping option:
        if self.nb_bootstrap_samples == 0:
            print(f'{filename}:: bootstrap resampling deactivated')
        elif (self.nb_bootstrap_samples > 0) and self.save_realizations:
            bsfilesdir = os.path.join(self.output_dir, 'bootstrap')
            if not os.path.isdir(bsfilesdir):
                os.mkdir(bsfilesdir)
            print(f'{filename}:: Output directory for bootstrap results is "{bsfilesdir}"')
        elif self.nb_bootstrap_samples < 0:
            raise ValueError(f'{filename}:: ERROR ! Number of bootstrap samples cannot be negative')


def unique_rows(a):  # Unused
    """
    Return only unique rows of a 2-D numpy array
    """
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def load_points(filename):
    """
    Load coordinates from input ASCII files.
    Several input format accepted:
    - Polygons: one vertex per line, two columns: lon, lat
    - Points/Epicenters: one epicenter per line, 4 or 8 columns
        * 4 columns format: floating date, lon, lat, magnitude
        * 8 columns format: floating date, lon, lat, mag, loc_unc_smaj_km, loc_unc_smin_km, loc_unc_az, mag_unc
    """
    def _check_polygon_validity(p):
        if not p.is_valid:
            raise ValueError('Polygon has a self-intersecting or invalid geometry')
        else:
            return p

    A = np.loadtxt(filename, comments='#')
    npts, ncols = A.shape
    if ncols == 2:
        # For polygon:
        print(f'{filename}:: Loaded polygon with {npts} vertices')
        geom = Polygon(A)
        _check_polygon_validity(geom)
        return geom

    elif ncols == 4:
        # For epicenters:
        # NB: small perturbations added to coordinates to avoid duplicates:
        print(f'{filename}:: Loaded {npts} epicenters (no uncertainties)')
        coords = A[:, 1:3]
        m = coords.mean(axis=0)
        xe, ye = 10 ** (np.round(np.log10(np.abs(m))) - 6)
        print(f'{filename}:: Apply minor perturbation to avoid duplicates (dX = {xe}; dY = {ye})')
        coords[:, 0] += xe * (1 - 2 * np.random.random(size=npts))
        coords[:, 1] += ye * (1 - 2 * np.random.random(size=npts))
        print(f'{filename}:: Longitude range = [{coords[:, 0].min()}; {coords[:, 0].max()}]')
        print(f'{filename}:: Latitude range = [{coords[:, 1].min()}; {coords[:, 1].max()}]')
        print(f'{filename}:: Magnitude range = [{A[:, 3].min()}; {A[:, 3].max()}]')
        print(f'{filename}:: Temporal range = [{A[:, 0].min()}; {A[:, 0].max()}]')
        geom = MultiPoint(coords)
        prepare(geom)  # In-place geom preparation for MultiPoint performance improvement
        return geom, A[:, 0], A[:, 3], None  # output: points, dates, mags

    elif ncols == 8:
        # For epicenters with uncertainties (loc, mag):
        # NB: small perturbations added to coordinates to avoid duplicates:
        print(f'{filename}:: Loaded {npts} epicenters with uncertainties (loc, mag)')
        coords = A[:, 1:3]
        uncert = {'loc_smaj': A[:, 4],  # Major semi-axis length in km for the ellipsoidal uncertainty
                  'loc_smin': A[:, 5],  # Minor semi-axis length in km for the ellipsoidal uncertainty
                  'loc_az': A[:, 6],  # Major semi-axis orientation from North, in degrees
                  'mag_unc': A[:, 7]  # Magnitude uncertainty
                  }
        m = coords.mean(axis=0)
        xe, ye = 10 ** (np.round(np.log10(np.abs(m))) - 6)
        print(f'{filename}:: Apply minor perturbation to avoid duplicates (dX = {xe}; dY = {ye})')
        coords[:,0] += xe * (1 - 2 * np.random.random(size=npts))
        coords[:,1] += ye * (1 - 2 * np.random.random(size=npts))
        print(f'{filename}:: Longitude range = [{coords[:,0].min()}; {coords[:,0].max()}]')
        print(f'{filename}:: Latitude range = [{coords[:,1].min()}; {coords[:,1].max()}]')
        print(f"{filename}:: Major semi-axis length range = [{uncert['loc_smaj'].min()}; {uncert['loc_smaj'].max()}] km")
        print(f"{filename}:: Minor semi-axis length range = [{uncert['loc_smin'].min()}; {uncert['loc_smin'].max()}] km")
        print(f"{filename}:: Major semi-axis azimuth range = [{uncert['loc_az'].min()}; {uncert['loc_az'].max()}] deg.")
        print(f'{filename}:: Magnitude range = [{A[:,3].min()}; {A[:,3].max()}]')
        print(f"{filename}:: Magnitude uncertainty range = [{uncert['mag_unc'].min()}; {uncert['mag_unc'].max()}]")
        print(f'{filename}:: Temporal range = [{A[:,0].min()}; {A[:,0].max()}]')
        geom = MultiPoint(coords)
        prepare(geom)  # In-place geom preparation for MultiPoint performance improvement
        return geom, A[:,0], A[:,3], uncert   # output: points, dates, mags

def load_bins(filename):
    """
    Load magnitude bin information
    """
    B = np.loadtxt(filename, comments='#')
    if len(B.shape) == 1:
        nb = 1
        B = B.reshape((1, B.shape[0]))
    else:
        nb = B.shape[0]
    print(f'{filename}:: Loaded {nb} bins')
    return B


def load_fmd_file(mbins_file, lons, lats, fmd_file=None, ibins=None, mmin=None, mmax=None, coord_precision=1E-6):
    """
    Load FMD information (Mmin, Mmax, and optionally bin durations)
    FMD_FILE Format: LON; LAT; MMIN; MMAX [; BIN_1_DURATION; ...; BIN_n_DURATION]
    Note:
        Missing MMax can be indicated as -9 or NaN in the input file.
    :param lons, lats: np.ndarray, cell longitudes and latitudes
    :param ibins: iterable, indices of magnitude-bins used in the analysis
    :param mmin: float, user-specified minimum magnitude truncation (overwrites values read from file)
    :param mmax: float, user-specified maximum magnitude truncation (overwrites values read from file)
    """
    ncells = len(lons)
    mbins = load_bins(mbins_file)
    if ibins is None:
        # By default, use all magnitude-bins:
        ibins = list(range(mbins.shape[0]))  # Bin indexes, NOT id's !
    nbins = len(ibins)
    mbins_durs = mbins[ibins, 4] - mbins[ibins, 3]  # Completeness durations for magnitude-bins

    mbins_durs_per_cell = np.ones((ncells, nbins))
    for i in range(ncells):
        # NB: bin list should normally be decimated to included only NBINS intervals:
        mbins_durs_per_cell[i, :] = mbins_durs

    cellinfo = np.ones((ncells, 4))  # For series of (lon, lat, mmin, mmax)
    if isinstance(fmd_file, str):
        print(f'>> Read limits (and optionally, durations) of freq.-mag. '
              + f'distributions in "{fmd_file}"')
        gridinfo = np.loadtxt(fmd_file, delimiter=';', comments='#')
        # Check GRIDINFO Format:
        # gridinfo[i,:] = [lon, lat, mmin, mmax, dur_i, dur_i+1, ..., dur_N]
        loaded_params = ''
        if gridinfo.shape[1] < 3:
            raise ValueError(f"{fmd_file}:: Missing columns (at least 3 required: lon, lat, mmin)")
        elif gridinfo.shape[1] == 3:
            print(f'{fmd_file}:: No max. magnitude specified, will use Mmax = Inf (untruncated model)')
            print(f'{fmd_file}:: No durations. Will use bin durations from {mbins_file}')
            loaded_params = 'Mmin'
        elif gridinfo.shape[1] == 4:
            print(f'{fmd_file}:: Will use max. magnitudes as specified in file')
            print(f'{fmd_file}:: No durations. Will use bin durations from {mbins_file}')
            loaded_params = 'Mmin, Mmax'
        elif gridinfo.shape[1] > 4:
            print(f'{fmd_file}:: Specified bin durations will supercede those given in "{mbins_file}"')
            loaded_params = 'Mmin, Mmax and bin durations'

        mmin_mesg_displayed = False
        for i in range(ncells):
            j = np.where((np.abs(gridinfo[:, 0] - lons[i]) < coord_precision) & \
                         (np.abs(gridinfo[:, 1] - lats[i]) < coord_precision) )[0]
            if len(j) == 0:
                print(f'{fmd_file}:: Cannot find cell with coordinates ' +
                      f'({lons[i]:.6f}; {lats[i]})')
                continue
            elif len(j) > 1:
                raise ValueError(f'{fmd_file}:: Found several lines with coordinates matching '
                               + f'({lons[i]:.6f}; {lats[i]}): {j}')

            cellinfo[i, :2] = gridinfo[j, :2]  # Copy (lon, lat) values

            # Manage Mmin values:
            if mmin is None:
                cellinfo[i, 2] = gridinfo[j, 2]  # Copy Mmin values
            else:
                if not mmin_mesg_displayed:
                    print(f'{fmd_file}:: Overwrite MMIN with the value given in command-line ({mmin:.2f})')
                    mmin_mesg_displayed = True
                cellinfo[i, 2] = mmin

            # Manage Mmax values:
            if mmax is None:
                if gridinfo.shape[1] > 3:
                    if (gridinfo[j, 3].item() == -9.0) or np.isnan(gridinfo[j, 3]):
                        cellinfo[i, 3] = np.inf  # untruncated G-R model
                    else:
                        cellinfo[i, 3] = gridinfo[j, 3].item()  # Copy Mmax
                else:
                    cellinfo[i, 3] = np.inf  # untruncated G-R model
            else:
                print(f'{fmd_file}:: Overwrite MMAX with the value given in command-line ({mmax:.2f})')
                cellinfo[i, 3] = mmax

            # Update bin durations, if available in gridinfo:
            if gridinfo.shape[1] > 4:
                mbins_durs_per_cell[i, :] = gridinfo[j, [4 + k for k in ibins]]
        print(f'{fmd_file}:: Loaded FMD parameters ({loaded_params}) for {ncells} cells')

    else:  # No file given
        print('>> Un-specified option "file_for_FMD_limits_and_durations" in configuration file')
        print('>> MMAX: No upper truncation (i.e., MMAX = inf.)')
        if mmin is None:
            mmin = min(mbins[ibins, 1])  # Use lower bound of magnitude intervals
            print(f'>> MMIN: Use the minimum lower bound of intervals read in "{mbins_file}" ({mmin:.2f})')
        else:
            print(f'>> MMIN: Use the value given in command-line for all pixels ({mmin:.2f})')

        if mmax is None:
            mmax = np.inf
            print(f'>> MMAX: Use an untruncated Gutenberg-Richter model')
        else:
            print(f'>> MMAX: Use the value given in command-line for all pixels ({mmax:.2f})')
        cellinfo[:, 0] = lons
        cellinfo[:, 1] = lats
        cellinfo[:, 2] = mmin
        cellinfo[:, 3] = mmax
    return cellinfo, mbins_durs_per_cell

def polygons_to_file(filename, polygons: GeometryCollection, zvalues=None, verbose=True):
    """
    Write polygons vertices to ASCII file formatted according to 
    the GMT format
    """
    n = len(polygons.geoms)
    with open(filename, 'wt') as fp:
        for i in range(n):
            if hasattr(zvalues, '__iter__'):  # Check if iterable
                fp.write(f'> Pol {i + 1:d}: -Z{zvalues[i]}\n')
            else:
                fp.write(f'> Pol {i + 1:d}: \n')
            for x, y in polygons.geoms[i].exterior.coords:
                fp.write(f'{x}  {y}\n')
    if verbose:
        print(f'{filename}:: {n} polygons written')


def change_zvalue_in_polygon_file(polygon_file: str, output_file: str, zvalues: np.ndarray):
    """
    Loads an ASCII polygon file formatted for GMT, as produced by method "lib.ioutils.polygon_to_file()",
    and duplicate file (in the same GMT format) containing updated Z-value with a different name.

    Args:
        polygon_file (str): Name of input polygon file (GMT format)
        output_file (str): Name of output polygon file (GMT format) with changed Z-values
        zvalues (np.ndarray): Modified Z-values, in cell-wise order 
        (i.e. zvalues[0] for "> Pol 0", zvalues[1] for "> Pol 1:" and so on...)
    """
    with open(polygon_file, 'rt') as fp:
        lines = fp.readlines()
    nl = len(lines)
    for i in range(nl):
        if lines[i].startswith('> Pol'):
            parts = lines[i].split('-Z')
            indx = int(parts[0].split(' ')[2].strip(':')) - 1  # Polygons indices go from 1 to N
            lines[i] = parts[0] + f'-Z{zvalues[indx]}\n'
    
    with open(output_file, 'wt') as fp:   
        fp.writelines(lines)
    print(f'>> Duplicated polygons and affected new Z-value in file {output_file}')
    

def minmax_in_polygon_file(polygon_file: str, exclude_inf=False):
    """
    Returns the minimum and maximum values of Z-values in the polygon file
    """
    zmin = np.inf
    zmax = -np.inf
    with open(polygon_file, 'rt') as fp:
        lines = fp.readlines()
    nl = len(lines)
    for i in range(nl):
        if lines[i].startswith('> Pol'):
            parts = lines[i].split('-Z')
            zvalue = float(parts[1])

            if zvalue < zmin:
                if exclude_inf and np.isfinite(zvalue):
                    zmin = zvalue
                elif not exclude_inf:
                    zmin = zvalue

            elif zvalue > zmax:
                if exclude_inf and np.isfinite(zvalue):
                    zmax = zvalue
                elif not exclude_inf:
                    zmax = zvalue

    return zmin, zmax    
        

def load_polygons(polygon_file: str):
    """
    Reads polygons from input file (in GMT ASCII format), returns a GeometryCollection
    object and the corresponding array of Z-values

    :param polygon_file: str, Path to file containing polygon coordinates in GMT ASCII format
    :return: (polygons: shapely.MultiPolygon object, zvalues: np.ndarray object)
    """
    with open(polygon_file, 'rt') as fp:
        lines = fp.readlines()
    nl = len(lines)
    polygons = list()
    zvalues = list()
    ip = -1
    for i in range(nl):
        if lines[i].startswith('#'):
            continue
        elif lines[i].startswith('>'):
            ip += 1
            if ip > 0:
                # If not the first polygon, flush the current list of coordinates in a new Polygon:
                polygons.append(Polygon(coords))
            coords = list()
            if lines[i].startswith('> Pol'): # Internal polygon numbering format
                parts = lines[i].split('-Z')
                zvalues.append(float(parts[1]))
            else:  # No polygon numbering
                zvalues.append(None)
        else:
            pt = Point([u for u in map(float, lines[i].rstrip().split())])  # Point((lon, lat))
            coords.append(pt)

    if ip > -1:
        # If no segment header ('>') found, flush the current list of coordinates in a new Polygon:
        polygons.append(Polygon(coords))

    return MultiPolygon(polygons), np.array(zvalues)


def load_grid(filename: str, scaling_factor=1.0):
    """
    Read grided density/count data from FILENAME
    """
    # Read used bin indices from header:
    with open(filename, 'rt') as fp:
        header = fp.readline()
    bin_ids = [int(e.replace(';','')) for e in header.split('bin_')[1:]]  # bin id's, NOT indices !
    # Load densities:
    values = np.loadtxt(filename, delimiter=';')
    # NB: Format of values[i,:] = [lon, lat, bin_i, bin_i+1, ..., bin_N]
    ncells = values.shape[0]
    nbins = values.shape[1] - 2
    # Scale rates:
    values[:, 2:] *= scaling_factor
    return values, ncells, nbins, bin_ids


def rates_to_csep(counts, bins_durations, mesh_step, magbins, verbose=True):
    """
    Format gridded rates data into the CSEP default gridded-forecast format.

    CSEP format is a tab-delimited ASCII file with the following columns for each
    space-magnitude bin:
    LON_0   LON_1   LAT_0   LAT_1   DEPTH_0 DEPTH_1 MAG_0   MAG_1   RATE    FLAG(=1 if used, 0 otherwise)
    Each row represents a single space-magnitude bin and the entire forecast file
    contains the rate for a specified time-horizon.

    :param counts, numpy.ndarray, one cell per line, with columns ordered as follows: formatted
        as counts[k,:] = [lon, lat, bin_i, bin_i+1, ..., bin_N]
    """
    ncells = counts.shape[0]
    nbins = magbins.shape[0]
    mmins = magbins[:, 1]
    mmaxs = magbins[:, 2]
    csep_array = np.zeros((ncells * nbins, 10))
    mindepth = 0
    maxdepth = 30
    flag = int(1)
    outputfile = 'csep_gridded_forecast.dat'
    k = -1  # Counter for valid cells
    nb = 0  # Number of valid bin (with specified durations)

    for i in range(ncells):
        # Check for valid durations and non-zero earthquake counts:
        if np.any(bins_durations[i, :] == -9) or (counts[i, 2:].sum() == 0.0):
            continue
        else:
            k += 1
            nb += nbins
        lon, lat = counts[i, 0:2]
        for j in range(nbins):
            csep_array[k * nbins + j, :] = np.array([lon - 0.5 * mesh_step,
                                         lon + 0.5 * mesh_step,
                                         lat - 0.5 * mesh_step,
                                         lat + 0.5 * mesh_step,
                                         mindepth,
                                         maxdepth,
                                         mmins[j],
                                         mmaxs[j],
                                         counts[i, 2 + j] / bins_durations[i, j],  # Seismic rate
                                         flag])

    # Update output array size (remove unused lines):
    csep_array = csep_array[:nb, :]
    np.savetxt(outputfile,
               csep_array,
               delimiter="\t",
               fmt=['%.3f']*8 + ['%.3g', '%d'])
    if verbose:
        print(f'{outputfile}:: {nb} space-magnitude bins written in CSEP format ({ncells * nbins - nb} bins without valid information on counts or durations)')


