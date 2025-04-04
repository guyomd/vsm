from importlib.metadata import version
import numpy as np
import pyproj
from shapely import (MultiLineString, 
                     Point,
                     MultiPoint, 
                     Polygon,
                     MultiPolygon, 
                     GeometryCollection, 
                     intersection,
                     voronoi_polygons,
                     prepare,
                     intersects,
                     make_valid,
                     get_num_points,
                     get_point,
                     is_closed,
                     )
from shapely.ops import transform, polygonize
from shapely.strtree import STRtree
from shapely.errors import GEOSException



## COORDINATE PROJECTION FUNCTIONS: ###########################################
def convert_to_EPSG(geom, in_epsg="EPSG:4326", out_epsg="EPSG=32618"):
    """
    Convert geographical coordinates in a shapely Geometry to
    another map projection
    """
    in_CRS = pyproj.CRS(in_epsg)
    out_CRS = pyproj.CRS(out_epsg)
    project = pyproj.Transformer.from_crs(in_CRS, out_CRS, always_xy=True).transform
    return transform(project, geom)


## GEOSPATIAL OPERATIONS: #####################################################
def eqdensity_per_polygon(polygons, weights: np.ndarray, scaling2unit=1.0, log_values=False):
    """
    Compute event density (per km^2) in each polygon, given individual weights in each polygon
    """
    values = []
    for pol, w in zip(polygons.geoms, weights):
        area = pol.area * (scaling2unit ** 2)
        density = w / area  
        if log_values:
            values.append(np.log10(density))
        else:
            values.append(density)
    return np.array(values)


def clipped_voronoi_diagram(multi_pt: MultiPoint, bounds: Polygon = None,
                            convertb4clip = None,
                            verbose=False):
    """
    Compute a clipped Voronoi diagram clipped within the 
    interior of the polygonal boundaries
    :param convertb4clip: dict, specify EPSG codes for internal conversion of Voronoi polygons 
        coordinates before clipping, so that conversion to an equal-area is guaranted before 
        intersecting polygons with the bounding polygon. 
        Syntax is: convertb4clipping = {'in_epsg': str, 'out_epsg': str}
    :param verbose: bool, increase verbosity when True (default: False)
    """
    # Compute Voronoi diagram:
    if len(multi_pt.geoms) == 1:
        # Special case when only 1 epicenter is provided in MULTI_PT:
        if convertb4clip:
            polygons = GeometryCollection(geoms=convert_to_EPSG(bounds,
                                                                in_epsg=convertb4clip['input_epsg'], 
                                                                out_epsg=convertb4clip['internal_epsg']))
        else:
            polygons = GeometryCollection(geoms=[bounds])
        prepare(polygons)
        weights = np.ones((1,))
        return polygons, weights
    
    else:
        # NB: Voronoi diagram must be computed using equal-area projection
        polygons = voronoi_polygons(multi_pt)

        if (bounds is not None) and (not polygons.is_empty):
            # If necessary, convert to the same CRS than bounds:
            if convertb4clip:
                polygons = convert_to_EPSG(polygons, 
                                           in_epsg=convertb4clip['internal_epsg'], 
                                           out_epsg=convertb4clip['input_epsg'])
            
            # Clip Voronoi polygons within the enclosing boundary polygon:
            clipped_geoms = []
            weights = []
            for p in polygons.geoms:
                if not p.is_valid:
                    # Simplify Polygon when it intersects itself:
                    if verbose:
                        print('WARNING: Make polygon valid in method "clipped_voronoi_diagram()"')
                        print(f'\tBEFORE: {p}')
                    p = make_valid(p) 
                    if verbose: 
                        print(f'\tAFTER: {p}')

                if intersects(p, bounds):
                    inter = intersection(p, bounds)
                    if isinstance(inter, Polygon):
                        clipped_geoms.append(inter)
                        weights.append(1.0)
                    elif isinstance(inter, MultiPolygon):
                        # Handles the situation when a Voronoi polygon is divided
                        # into several polygons after intersection,
                        # then distribute the unit weight (or count) over all sub-polygons:
                        nsub = len(inter.geoms)
                        if verbose:
                            print(f'WARNING: Voronoi polygon sub-divided into {nsub} parts after intersection:')
                            print(f'\t{inter}')
                        areas = np.array([inter.geoms[i].area for i in range(nsub)])
                        total_area = areas.sum()
                        for i in range(nsub):
                            clipped_geoms.append(inter.geoms[i])
                            weights.append(areas[i] / total_area)
                elif verbose:
                    print(f'WARNING: Skipping polygon --> Empty intersection with bounds:\t{p}')
            weights = np.array(weights)
            clipped_geoms = GeometryCollection(geoms=clipped_geoms)
            
            # If necessary, convert backwards:
            if convertb4clip:
                clipped_geoms = convert_to_EPSG(clipped_geoms, 
                                                in_epsg=convertb4clip['input_epsg'], 
                                                out_epsg=convertb4clip['internal_epsg'])
                clipped_geoms = GeometryCollection(geoms=clipped_geoms)
            prepare(clipped_geoms)  # In-place geom preparation for performance improvement
            return clipped_geoms, weights
        
        else:
            prepare(polygons)  # In-place geom preparation for performance improvement
            weights = np.ones((len(polygons.geoms),)) 
            return polygons, weights


def build_mesh(bounds: Polygon, step: float, scaling2unit: float = 1.0):
    xmin, ymin, xmax, ymax = bounds.bounds
    # NB: Mesh of cells must encompass completely the bounding polygon, so add one
    #     bin step in every direction. Errors may occur is mesh does not extend
    #     sufficiently away from the bounding polygon, due to rounding errors
    #     associated with coordinate conversion.
    step *= scaling2unit
    lons = np.arange(xmin - step, xmax + 2 * step, step)
    lats = np.arange(ymin - step, ymax + 2 * step, step)
    nlon = len(lons)
    nlat = len(lats)
    print(f'>> Initial mesh....')
    print(f'>> West-East bounds ({nlon} cells): [{lons.min()}; {lons.max() + step}]')
    print(f'>> South-North bounds ({nlat} cells): [{lats.min()}; {lats.max() + step}]')
    cells = []
    for lon in lons:
        for lat in lats:
            cells.append(Polygon([Point(lon, lat),
                                  Point(lon + step, lat),
                                  Point(lon + step, lat + step),
                                  Point(lon, lat + step)]))
    cells = GeometryCollection(cells)
    # Keep only cells included within the bounding polygon:
    extended_bounds = bounds.buffer(0.5 * step, cap_style='square')
    cells_inside = GeometryCollection([g for g in cells.geoms if intersects(g, extended_bounds)])
    del cells
    prepare(cells_inside)
    centroids = np.array([(p.centroid.x, p.centroid.y) for p in cells_inside.geoms])
    print(f'>> Number of cells inside bounding polygon: {len(cells_inside.geoms)}')
    return cells_inside, centroids


def get_multi_index_in_polygon(polygon, multigeom_or_tree, geoms=None):
    """
    Returns the indices of elements in a MultiGeometry object (i.e. MultiPoint, MultiPolygon)
    intersecting a Polygon object

    :param polygon: instance of shapely.geometry.Polygon object
    :param multigeom_or_tree: instance of classes shapely.geometry.MultiPoint/MultiPolygon or shapely.strtree.STRtree
    :param geoms: list of geometries indexed in the R-tree
    :return: a list of indices
    """

    if isinstance(multigeom_or_tree, STRtree) and (geoms is not None):
        tree = multigeom_or_tree
        mg = geoms
    else:
        mg = list(multigeom_or_tree.geoms)
        tree = STRtree(mg)

    if int(version('shapely').split('.')[0]) < 2:
        index_by_id = dict((id(p), i) for i, p in enumerate(mg))
        indices = np.array(
            [index_by_id[id(p)] for p in tree.query(polygon)])  # only valid for shapely version < 2.0
    else:
        indices = tree.query(polygon)  # only valid for shapely version >= 2.0.
        # Warning! It appears that the list of indices returned differs slightly between version < and >= 2...
        #          --> Is it because STRtree is not defined in the same manner (ordering of polygons) ?
    return indices


def eqcounts_per_cell(cells_m, vor_diagram_m, weights: np.ndarray, scaling2unit=1.0, verbose=False):
    """
    Computes (floating) earthquake count and density per cell by summing 
    the quantities proportional to the intersecting areas of all Voronoi 
    polygons with each cell

    :param cells_m: shapely.MultiPolygon instance, with coordinates 
        expressed in a equal-area CRS
    :param vor_diagram_m: shapely.GeometryCollection instance, list of
        Voronoi polygons, expressed in a equal-area CRS
    :param weights: np.ndarray, earthquake weights for each Voronoi polygon (<=1.0)
    :param scaling2unit: float, scaling coefficient to convert areas in km^2
    :param verbose: bool, set verbosity mode (default: False).
    """
    counts = []
    densities = []
    if len(vor_diagram_m.geoms) == 0:
        raise AssertionError(f'No event included into the mesh domain')
    if vor_diagram_m.envelope.within(cells_m.envelope) == False:
        raise AssertionError(f'Clipped Voronoi diagram is not fully included into the mesh domain')
    vor_geoms_m = list(vor_diagram_m.geoms)
    tree = STRtree(vor_geoms_m)
    for cell in cells_m.geoms:
        count = 0
        indices = get_multi_index_in_polygon(cell, tree, vor_geoms_m)
        for ip in indices:
            # Determines the intersecting area between mesh square and each polygon:
            vor_polygon = vor_diagram_m.geoms[ip]
            if not vor_polygon.is_valid:
                if verbose:
                    print('INFO: Make polygon valid in method "eqcounts_per_cell()"')
                    print(f'\tBEFORE: {vor_polygon}')
                vor_polygon = make_valid(vor_polygon)
                if verbose: 
                    print(f'\tAFTER: {vor_polygon}')
            inter = cell.intersection(vor_polygon, grid_size=0.000001)
            # Increment event count for current cell:
            count += weights[ip] * inter.area / vor_polygon.area  # Ratio of areas must be expressed in m^2
        counts.append(count)
        densities.append(count / (cell.area * (scaling2unit ** 2)))
    return np.array(counts), np.array(densities)  # Cell densities


def select_events(mp_epic: MultiPoint, mags: np.ndarray, dates: np.ndarray, polygon: Polygon, magbin: np.ndarray):
    """
    Return a subset of input epicenters MP_EPIC that are located within a given 
    POLYGON, with magnitude included in a given magnitude bin and occurrence time
    included between starting and ending years for the specified magnitude bin.
    :param mp_epic: shapely.MultiPoint, earthquake epicenters
    :param mags: numpy.ndarray, earthquake magnitudes
    :param dates: numpy.ndarray, earthquake occurrence times (formatted as floats)
    :param polygon: shapely.Polygon, geographical bounds for event selection
    :param magbin: numpy.ndarray, properties of a magnitude bin, formatted as
        magbin = [index, mmin, mmax, tstart, tend]
    """
    inpolygon = np.array([polygon.contains(point) for point in mp_epic.geoms])
    isselected = np.logical_and.reduce((inpolygon, 
                                        mags >= magbin[1], 
                                        mags < magbin[2],
                                        dates >= magbin[3],
                                        dates <= magbin[4]))
    subset = MultiPoint([point for point, inselection in zip(mp_epic.geoms, isselected) if inselection])
    return subset, mags[isselected], dates[isselected]


def interpolate_polygon_coords(polygon: Polygon, n=1000):
    """Returns a polygon with the same shape but with more finely discretized 
       edges. The returned polygon has a higher density of coordinates, with N
       points evenly distributed along its exterior edge.

    Args:
        polygon : shapely.Polygon, input Polygon
        
    """
    dstep = 1 / (n - 1)
    pts = [polygon.exterior.interpolate(d, normalized=True) for d in np.arange(0, 1 + dstep, dstep)]
    interpolated_polygon = Polygon(pts)
    prepare(interpolated_polygon)
    return interpolated_polygon


def random_locations_from_ellipsoid(x_km, y_km, semi_major_axis_len, semi_minor_axis_len,
                                         azim_major_axis, n=1, rng=np.random.default_rng()):
    """
    Draw N random event locations centered on (x, y) within an 2-D gaussian error ellipsoid parameterized by
    its major and minor semi-axis length and by the azimuth of its main axis.

    :param x_km: float, ellipsoid (or event) central longitude
    :param y_km: float, ellipsoid (or event) central longitude
    :param semi_major_axis_len: float, semi-major axis length of the ellipsoid, in km
    :param semi_minor_axis_len: float, semi-minor axis length of the ellipsoid, in km
    :param azim_major_axis: float, azimuth of the ellipsoid major axis, in degrees from North to East
    :param n: int, number of random samples
    :return: (x, y), random event coordinates where x and y are nd.ndarray with shape (n,)
    """
    means = np.squeeze(np.array((x_km, y_km)))
    smax2 = semi_major_axis_len ** 2
    smin2 = semi_minor_axis_len ** 2
    az_radians = azim_major_axis * np.pi / 180.0
    varx = (np.sin(az_radians) ** 2) * smax2 + (np.cos(az_radians) ** 2) * smin2
    vary = (np.cos(az_radians) ** 2) * smax2 + (np.sin(az_radians) ** 2) * smin2
    cov_xy = (smax2 - smin2) * np.cos(az_radians) * np.sin(az_radians)
    covmat = np.array([[varx, cov_xy], [cov_xy, vary]])  # Covariance matrix for the 2-D ellipsoid
    #corrcoef = cov_xy / np.sqrt(varx * vary)  # Correlation coefficient
    rv = rng.multivariate_normal(means, covmat, size=n)
    return rv[:, 0], rv[:, 1]


def reorder_germs(diagram: GeometryCollection, germs: MultiPoint):
    """
    Reorder germs in the same order than Voronoi polygon.
    Set germ to None when no germ can be associated with a polygon.

    :param diagram: shapely.GeometryCollection instance, Voronoi diagram
    :param germs: shapely.MultiPoint instance, Germs of Voronoi polygons
    """
    ordered_germs = list()
    for polygon in diagram.geoms:
        germ = None
        for pt in germs.geoms:
            if pt.within(polygon):
                germ = pt
                break
        ordered_germs.append(germ)
    return ordered_germs


def polygon2triangles(polygon: Polygon, germ: Point):
    """
    Subdivide a polygon with N vertices into a collection of triangles all sharing the input germ as a common vertex,
    and using each two consecutive vertices of the input polygon

    :param polygon: shapely.Polygon instance, Voronoi polygon
    :param germ: shapely.Point instance, Germ of the Voronoi polygon
    """
    if is_closed(polygon.exterior):
        ntri = get_num_points(polygon.exterior) - 1  # Do not account for repeated last point
    else:
        ntri = get_num_points(polygon.exterior)
    triangles = list()
    weights = list()
    for i in range(ntri):
        if i == ntri - 1:
            tripts = get_point(polygon.exterior, [i, 0]).tolist()
        else:
            tripts = get_point(polygon.exterior, [i, i + 1]).tolist()
        tripts.append(germ)
        triangles.append(Polygon(tripts))
        weights.append(1.0 / ntri)  # Distribute uniform weight over each of the NTRI sub-triangles
    return triangles, np.array(weights)


def subdivide_voronoi_cells(diagram: GeometryCollection, weights: np.ndarray, germs: MultiPoint):
    """
    Returns a MultiPolygon object consisting in a collection of Voronoi cells subdivided into triangles

    :param diagram: shapely.GeometryCollection instance, Voronoi diagram
    :param weights: numpy.ndarray instance, earthquake count for each Voronoi polygon (can be less than 1 if polygon was
    split before)
    :param germs: shapely.MultiPoint instance, Germs of Voronoi polygons. Must be in the same order than DIAGRAM!
    """
    subdivided_diagram = list()
    subdivided_weights = list()
    for polygon, germ, w in zip(diagram.geoms, germs.geoms, weights):
        if germ is None:
            # May happen when Voronoi polygon were split before, in this case cancel subdivision for the polygon:
            subdivided_diagram.append(polygon)
            subdivided_weights.append(w)
        else:
            triangles, triweights = polygon2triangles(polygon, germ)
            subdivided_diagram.append(triangles)
            subdivided_weights.append(triweights * w)
    return GeometryCollection(subdivided_diagram), np.array(subdivided_weights)

