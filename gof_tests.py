import os
import numpy as np
import shapely
from scipy.stats import norm, hmean
from argparse import ArgumentParser
from tqdm import tqdm
import pygmt
from shapely import (intersection,
                     Point,
                     MultiPoint,
                     Polygon,
                     MultiPolygon)

from lib.ioutils import (ParameterSet,
                         load_points,
                         load_bins)
from lib.geoutils import convert_to_EPSG




class SeismicityRateMap():
    def __init__(self, prms, mmin, ab_file=None):
        """
        :param prms: instance of lib.ioutils.ParameterSet class
        :param mmin: float, minimum magnitude used for the computation of rates
        :param ab_file: str, path to the file containing (a, b) estimates, e.g. 'ab_values.txt'
        """
        self.mmin = mmin
        self.prms = prms
        self.cells = None
        self.rates = None
        if ab_file is not None:
            self.build(ab_file)

    def _define_cell(self, lon, lat):
        half_step = 0.5 * self.prms.mesh_step
        cell = Polygon([
            Point((lon - half_step, lat - half_step)),
            Point((lon - half_step, lat + half_step)),
            Point((lon + half_step, lat + half_step)),
            Point((lon + half_step, lat - half_step)),
            Point((lon - half_step, lat - half_step))
        ])
        return cell

    def _compute_rate_in_cell(self, a, b, cell):
        # Compute cel area:
        cell_m = convert_to_EPSG(cell,
                                 in_epsg=self.prms.input_epsg,
                                 out_epsg=self.prms.internal_epsg)
        cell_area = cell_m.area * (self.prms.epsg_scaling2km ** 2)  # Convert to km^2
        scaling_factor = cell_area / self.prms.density_scaling_factor
        rate_ge_mmin = 10 ** (a - b * self.mmin) * scaling_factor
        return rate_ge_mmin, cell_area

    def build(self, ab_file):
        """
        Compute seismicity rate in each cell of the target area based on
        parameters (a, b) of the Gotenberg-Richter relationship
        :param ab_file: str, path to the file containing (a, b) estimates, e.g. 'ab_values.txt'.
            File format: 1 cell per line, with columns ordered as (lon; lat; a; b; da; db; rho_ab; mc)
        :return:
        """
        ab_basename = os.path.basename(ab_file)
        ab_values = np.loadtxt(ab_file, delimiter=';', skiprows=1)
        n = ab_values.shape[0]
        cell_list = list()
        self.rates = list()
        self.cell_areas = list()
        for i in range(n):
            lon, lat, a, b, da, db, rho_ab, mc = ab_values[i, :]

            if np.isnan(a) or np.isinf(a):
                continue

            if mc > self.mmin:
                print(f'Warning! ({ab_basename}, l.{i+1}) ' +
                      f'higher completeness threshold ({mc:.1f}) ' +
                      f'than required Mmin ({self.mmin:.1f})')

            cell = self._define_cell(lon, lat)
            cell_list.append(cell)
            rate, area = self._compute_rate_in_cell(a, b, cell)
            self.rates.append(rate)
            self.cell_areas.append(area)
        self.cells = MultiPolygon(cell_list)
        self.rates = np.array(self.rates)
        self.cell_areas = np.array(self.cell_areas)
        print(f'{os.path.basename(ab_file)}:: Rates computed for {len(self.cells.geoms)} valid cells')


class ResidualAnalysisTester():
    def __init__(self, prms, map, epicenters, tmin, tmax):
        """
        Implements a super-thining of the spatial point process as described in Clemens et al. (2012).

        Ref: Clemens, R.A., Schoenberg, F. P. and Veen, A., 2012, Evaluation of space-time point process models
        using super-thinning, Environmetrics, 23, 606-616. http://doi.org/10.1002/env.2168

        !param prms, instance of lib.ioutils.ParameterSet class
        :param map: instance of SeismicityRateMap class
        :param epicenters: shapely.MultiPoint class
        """
        self.epic = epicenters
        self.k = np.median(map.rates)  # Tuning parameter
        self.map = map
        self.prms = prms
        self.tmin = tmin
        self.tmax = tmax
        self.rng = np.random.default_rng()  # Random number generator

    def _thin_point_process(self, k):
        thinned_epicenters = list()
        for cell, rate in zip(self.map.cells.geoms, self.map.rates):
            epc = self._find_epicenters_in_cell(cell, self.epic)
            if epc is None:
                continue
            n = len(epc.geoms)
            p = min(k / rate, 1.0)
            u = self.rng.uniform(size=n)
            thinned_epicenters += [epc.geoms[i] for i in range(n) if (u[i] <= p)]
        return MultiPoint(thinned_epicenters)

    def _generate_Cox_process(self, k):
        cox_process = list()
        for cell, rate in zip(self.map.cells.geoms, self.map.rates):
            t = self._poisson_timeseries(k, t0=self.tmin, tmax=self.tmax)
            n = len(t)
            epc = self._random_epicenters_in_cell(cell, n)
            p = max((k - rate) / k, 0)
            u = self.rng.uniform(size=n)
            cox_process += [epc.geoms[i] for i in range(n) if (u[i] <= p)]
        return MultiPoint(cox_process)

    def _poisson_timeseries(self, rate, n=None, tmax=None, t0=0):
        """
        Generate a synthetic time series for a stationary Poisson process.
        Either a total number of samples (N) or a maximum event occurrence time (tmax)
        must be specified.

        :param rate: float, event occurrence rate
        :param n: int, number of synthetic times of occurrence
        :param tmax: float, maximum event time occurrence allowed
        :param t0: float, initial event occurrence time
        """
        if ((n is None) and (tmax is None)) or ((n is not None) and (tmax is not None)):
            raise ValueError(f'Either N or TMAX must be specified, but not both!')
        if tmax is None:
            x = self.rng.random((n,))
            dt = -np.log(x) / rate
            t = t0 + np.cumsum(dt)
        elif n is None:
            last_t = t0
            t = []
            while (last_t < tmax) and (rate > 0):
                t.append(last_t)
                x = self.rng.random()
                dt = -np.log(x) / rate
                last_t = t[-1] + dt
        return t

    def _find_epicenters_in_cell(self, cell, epicenters):
        inter = intersection(cell, epicenters)
        if not shapely.is_empty(inter):
            if isinstance(inter, Point):
                # Single-point case, reformat as MultiPoint!
                inter = MultiPoint([inter])
            return inter  # shapely.MultiPoint object
        else:
            return None

    def _random_epicenters_in_cell(self, cell, n):
        xmin, ymin, xmax, ymax = cell.bounds
        xs = self.rng.uniform(size=n) * (xmax - xmin) + xmin
        ys = self.rng.uniform(size=n) * (ymax - ymin) + ymin
        epc = MultiPoint([Point(x, y) for x, y in zip(xs, ys)])
        return epc

    def _find_matching_cell_and_rate(self, pt):
        nc = len(self.map.cells.geoms)
        for i in range(nc):
            if pt.within(map.cells.geoms[i]):
                return i, self.map.rates[i]

    def _Kw_function(self, r, xy, rates):
        """
        Compute an estimator of the weighted Ripley's K-function Kw, as defined in Veen and Schoenberg (2006)

        Ref: Veen, A., and Schoenberg, F. P., 2006, Assessing spatial point process models using weighted K-functions:
        analysis of California earthquakes, in: Case studies in spatial point process modeling, Lecture notes in
        Statistics, 185, pp. 293-306, Springer, New-York.
        """
        n = xy.shape[0]
        a = self.map.cell_areas.sum()  # km^2
        l_star = rates.min()
        Kw = 1 / ((l_star ** 2) * a)
        counts = 0
        for i in range(n):
            xi = xy[i, 0]
            yi = xy[i, 1]
            wi = l_star / rates[i]
            for j in range(n):
                xj = xy[j, 0]
                yj = xy[j, 1]
                wj = l_star / rates[j]
                if (i != j):
                    dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                    if (dist <= r):
                        counts += wi * wj
        Kw *= counts
        asympt = norm( np.pi * (r ** 2), np.sqrt(2 * np.pi * (r ** 2) / (a * hmean(rates) ** 2)) )
        return Kw, asympt.ppf(0.025), asympt.ppf(0.975)

    def _centered_Lw_function(self, rvalues, mpts):
        rates = np.array([self._find_matching_cell_and_rate(pt)[1] for pt in mpts.geoms])
        rates *= self.tmax - self.tmin
        mpts_m = convert_to_EPSG(mpts,
                                 in_epsg=self.prms.input_epsg,
                                 out_epsg=self.prms.internal_epsg)
        xy = np.array([[pt.x * self.prms.epsg_scaling2km, pt.y * self.prms.epsg_scaling2km] for pt in mpts_m.geoms])
        Lw = np.zeros_like(rvalues)
        Lw95low = np.zeros_like(rvalues)
        Lw95upp = np.zeros_like(rvalues)
        for i in tqdm(range(len(rvalues))):
            r = rvalues[i]
            Kw, Kw95low, Kw95upp = self._Kw_function(r, xy, rates)
            Lw[i] = np.sqrt(Kw / np.pi) - r
            Lw95low[i] = np.sqrt(Kw95low / np.pi) - r
            Lw95upp[i] = np.sqrt(Kw95upp / np.pi) - r
        return Lw, Lw95low, Lw95upp

    def run(self):
        # First, apply thinning of the space-time point process:
        z1 = self._thin_point_process(self.k)
        print(f'Nevents in original catalogue: {len(self.epic.geoms)}')
        print(f'Nevents in Z1: {len(z1.geoms)}')
        # Then, create a Cox process directed by max{self.k - rate(x,t), 0):
        z2 = self._generate_Cox_process(self.k)
        print(f'Nevents in Z2: {len(z2.geoms)}')
        # Superpos both processes:
        res_pts = [geom for geom in z1.geoms] + [geom for geom in z2.geoms]
        print(f'>> {len(res_pts)} super-thinned residual points obtained')
        z = MultiPoint(res_pts)   # Super-thinned residual points

        # Now, inspect Z for uniformity using the centered Lw-function:
        r = np.arange(0.1, 100, 5.0)
        Lw, Lw95low, Lw95upp = self._centered_Lw_function(r,z)
        return r, Lw, Lw95low, Lw95upp

    def plot(self, r, Lw, Lw95low, Lw95upp):
        frame = ["WSne", f"xaf+lkm", f"yaf+lLw(r) - r"]
        bounds = [min(r), max(r), min(Lw.min(), Lw95low.min()) * 2, max(Lw.max(), Lw95upp.max()) * 2]
        fig = pygmt.Figure()
        fig.basemap(region=bounds,
                    projection="X12c/12c",
                    frame=frame)
        fig.plot(x=r,
                 y=Lw,
                 pen='1p,black')
        fig.plot(x=r,
                 y=Lw95low,
                 pen='1p,black,dashed')
        fig.plot(x=r,
                 y=Lw95upp,
                 pen='1p,black,dashed',
                 label='95%% confidence interval')
        fig.legend()
        figname = 'residuals_homogeneity.png'
        fig.savefig(figname, transparent=True)
        print(f'>> Figure saved in "{figname}"')



if __name__ == "__main__":

    # Read input arguments:
    parser = ArgumentParser(
        description="Compute Goodness-of-fit tests, based on the comparison between the model and the input catalogue")

    parser.add_argument("configfile",
                        nargs='?',
                        default="parameters.txt",
                        help="Configuration file")

    parser.add_argument("--mmin",
                        help="Set the minimum magnitude used for Goodness-of-fit tests",
                        default=None,
                        type=float)

    args = parser.parse_args()

    # Load parameters:
    prms = ParameterSet()
    prms.load_settings(args.configfile)
    ab_file = os.path.join(prms.output_dir, 'ab_values.txt')

    if args.mmin is None:
        # Define the minimum magnitude from the smallest bin:
        mbins = load_bins(prms.bins_file)
        mmin = mbins[0, 1]
        print(f'>> Will use minimum magnitude from the smallest bin in {prms.bins_file}: {mmin}')
    else:
        mmin = args.mmin

    # Define variables for Goodness-of-fit tests:
    map = SeismicityRateMap(prms, mmin, ab_file)
    points, dates, mags, weights, _ = load_points(prms.epicenters_file)
    epicenters = MultiPoint([pt for pt, mag in zip(points.geoms, mags) if (mag >= mmin)])

    # Run tests:
    tester = ResidualAnalysisTester(prms, map, epicenters, min(dates), max(dates))
    r, Lw, Lw95low, Lw95upp = tester.run()
    tester.plot(r, Lw, Lw95low, Lw95upp)
