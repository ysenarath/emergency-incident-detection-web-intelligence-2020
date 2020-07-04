from functools import partial

import pyproj
from shapely.geometry import Point
from shapely.ops import transform


def project_coordinates(lng, lat, radius=None):
    local_azimuthal_projection = "+proj=aeqd +R=6371000 +units=m +lat_0={} +lon_0={}".format(lat, lng)
    wgs84_to_aeqd = partial(
        pyproj.transform,
        pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"),
        pyproj.Proj(local_azimuthal_projection),
    )
    aeqd_to_wgs84 = partial(
        pyproj.transform,
        pyproj.Proj(local_azimuthal_projection),
        pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"),
    )
    center = Point(float(lng), float(lat))
    point_transformed = transform(wgs84_to_aeqd, center)
    if radius is not None:
        buffer = point_transformed.buffer(int(radius))  # in meters
    else:
        buffer = point_transformed
    # Get the polygon with lat lon coordinates
    return transform(aeqd_to_wgs84, buffer)
