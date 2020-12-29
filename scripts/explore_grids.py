import math
from functools import partial

import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import streamlit as st
from rtree import index
from shapely.geometry import Point, MultiPoint
from shapely.ops import cascaded_union
from shapely.ops import transform

idx = index.Index()

st.title('Grid Clustering for WAZE.')

incident_df = pd.read_pickle('output/tdot_12_incidents.pkl')
incident_df['time'] = pd.to_datetime(incident_df['timestamp'], unit='s')
st.write('Accident Incident Data: ')
st.write(incident_df)

report_df = pd.read_csv('output/waze_12_etrims.csv', index_col=0)
# pubMillis (NUMERIC) - milliseconds since epoch
report_df['time'] = pd.to_datetime(report_df['pubMillis'], unit='ms')
dist_to_segment = st.slider('Max Distance to Segment (in degrees / 1000)', 0, 20, 1)  # distance in degrees
report_df = report_df[report_df.seg_id_dist * 1000 < dist_to_segment]
st.write('Accident Report Data: ')
st.write(report_df)

grid_size_resolution = []

for resolution in range(16):
    h3_grids = [h3.geo_to_h3(lat, lng, resolution) for lat, lng in zip(incident_df.lat, incident_df.lng)]
    grid_size_resolution.append({
        'resolution': resolution,
        'len_h3_grids': len(np.unique(h3_grids)),
    })

grid_size_resolution_df = pd.DataFrame(grid_size_resolution)

st.write('See https://h3geo.org/docs/core-library/restable for resolution sizes.')

st.line_chart(grid_size_resolution_df)

# PLOT GRIDS

aperture_size = st.slider('Select the Grid Resolution', 0, 15, 7)

hex_col = 'hex' + str(aperture_size)

# find hexs containing the points
incident_df[hex_col] = incident_df.apply(lambda x: h3.geo_to_h3(x.lat, x.lng, aperture_size), 1)

# aggregate the points
incident_g_df = incident_df.groupby(hex_col).size().to_frame('cnt').reset_index()

# find center of hex for visualization
incident_g_df['lat'] = incident_g_df[hex_col].apply(lambda x: h3.h3_to_geo(x)[0])
incident_g_df['lng'] = incident_g_df[hex_col].apply(lambda x: h3.h3_to_geo(x)[1])


# Functions
def plot_scatter(df, metric_col, x='lng', y='lat', marker='.', alpha=1, figsize=(16, 12), colormap='viridis'):
    df.plot.scatter(x=x, y=y, c=metric_col, title=metric_col
                    , edgecolors='none', colormap=colormap, marker=marker, alpha=alpha, figsize=figsize)
    plt.xticks([], [])
    plt.yticks([], [])


# pltot the hexs
plot_scatter(incident_g_df, metric_col='cnt', marker='o', figsize=(16, 12))
plt.title('hex-grid: accidents (incident)')

st.pyplot()

grid_cells = [MultiPoint(h3.h3_to_geo_boundary(h)).bounds for h in np.unique(incident_df[hex_col])]

for pos, cell_bounds in enumerate(grid_cells):
    idx.insert(pos, cell_bounds)

radius = 500  # in meters

# Obtain polygons for each WAZE report
polygons = []
progress_bar = st.progress(0)
progress_bar_len = report_df.shape[0]
progress_bar_count = 0.0
for lat, lng in zip(report_df.lat, report_df.lng):
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
    buffer = point_transformed.buffer(radius)
    # Get the polygon with lat lon coordinates
    circle_poly = transform(aeqd_to_wgs84, buffer)
    polygons.append(circle_poly)
    progress_bar_count += 1
    progress_bar.progress(math.floor(progress_bar_count * 100 / progress_bar_len))

# Loop through each Shapely polygon
progress_bar = st.progress(0)
progress_bar_len = len(polygons)
progress_bar_count = 0.0
for poly in polygons:
    # Merge cells that have overlapping bounding boxes
    merged_cells = cascaded_union([grid_cells[pos] for pos in idx.intersection(poly.bounds)])
    # Now do actual intersection
    # print(poly.intersection(merged_cells).area)
    progress_bar_count += 1
    progress_bar.progress(math.floor(progress_bar_count * 100 / progress_bar_len))
