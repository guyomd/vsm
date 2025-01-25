import os
import numpy as np
from argparse import ArgumentParser
from lib.ioutils import (ParameterSet,
                         load_bins,
                         load_grid,
                         load_fmd_file,
                         rates_to_csep)


if __name__ == "__main__":

    # Read input arguments:
    parser = ArgumentParser(
        description="Produce a grid of earthquake rates formatted according to the CSEP Gridded Forecast format")

    parser.add_argument("configfile",
                        nargs='?',
                        default="parameters.txt",
                        help="Configuration file")

    args = parser.parse_args()

    # Load parameters:
    prms = ParameterSet()
    prms.load_settings(args.configfile)

    # Load counts:
    counts_file = os.path.join(prms.output_dir, 'gridded_counts.txt')
    counts, ncells, nbins, bin_ids = load_grid(counts_file, scaling_factor=1.0)

    # Load magnitude bins and keep only those mentionned in gridded counts:
    mbins = load_bins(prms.bins_file)
    ibins = [k for k, bin_id in enumerate(mbins[:, 0]) if int(bin_id) in bin_ids]
    mbins = mbins[ibins, :]

    # Load magnitude-bins durations:
    cellinfo, bins_durations_per_cell = load_fmd_file(
        prms.bins_file,
        counts[:, 0],  # Longitudes
        counts[:, 1],  # Latitudes
        fmd_file=prms.fmd_info_file,
        ibins=ibins,
        mmin=None)
    print(f'counts: min = {counts[:, 2:].min()}   max = {counts[:, 2:].max()}')
    print(f'bin_durations_per_cell: min = {bins_durations_per_cell.min()}   max = {bins_durations_per_cell.max()}')
    rates_to_csep(counts, bins_durations_per_cell, prms.mesh_step, mbins)



