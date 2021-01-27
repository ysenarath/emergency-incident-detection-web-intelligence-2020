#!/usr/bin/env python
# coding: utf-8

# In[144]:


from datetime import timedelta
from functools import partial

import geopandas
import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import tqdm
from shapely.geometry import Point, Polygon
from shapely.ops import transform
from shapely.strtree import STRtree

# In[2]:


incident_df = pd.read_pickle('../output/tdot_12_incidents.pkl')
incident_df['time'] = pd.to_datetime(incident_df['timestamp'], unit='s')

# In[3]:


report_df = pd.read_csv('../output/waze_12_etrims.csv', index_col=0)
# pubMillis (NUMERIC) - milliseconds since epoch
report_df['time'] = pd.to_datetime(report_df['pubMillis'], unit='ms')
dist_to_segment = 1  # distance in degrees
report_df = report_df[report_df.seg_id_dist * 1000 < dist_to_segment]

# In[4]:


grid_size_resolution = []

for resolution in range(16):
    h3_grids = [h3.geo_to_h3(lat, lng, resolution) for lat, lng in zip(incident_df.lat, incident_df.lng)]
    grid_size_resolution.append({
        'resolution': resolution,
        'len_h3_grids': len(np.unique(h3_grids)),
    })

grid_size_resolution_df = pd.DataFrame(grid_size_resolution)

grid_size_resolution_df.plot.line()


# In[10]:


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
        buffer = point_transformed.buffer(radius)  # in meters
    else:
        buffer = point_transformed
    # Get the polygon with lat lon coordinates
    return transform(aeqd_to_wgs84, buffer)


# In[424]:


# Example grid - incident overlap
lng, lat = -122.431297, 37.773972  # lon lat for San Francisco (from StackOverflow)

radius = 1000  # in m

aperture_size = 6  # Grid Resolution

buffer = project_coordinates(lng, lat, radius)

print(buffer.bounds)

pt = project_coordinates(lng, lat)

print((pt.y, pt.x), end='\n\n')

boundary = h3.h3_to_geo_boundary(h3.geo_to_h3(pt.y, pt.x, aperture_size))

print(boundary, end='\n\n')

grid = Polygon([(x, y) for (y, x) in boundary])

print(grid.bounds, end='\n\n')

print(grid.intersection(buffer).area, end='\n\n')

df1 = geopandas.GeoDataFrame({'geometry': buffer, 'df1': [1, 2]})

df2 = geopandas.GeoDataFrame({'geometry': grid, 'df2': [1, 2]})

ax = df1.plot(color='red');

df2.plot(ax=ax, color='green', alpha=0.5);

# In[356]:


waze_report_uncertainty_regions = []
for lng, lat in tqdm.tqdm(list(zip(report_df.lng, report_df.lat))):
    waze_report_uncertainty_regions.append(project_coordinates(lng, lat, radius))


# In[425]:


# Functions
def plot_scatter(df, metric_col, x='lng', y='lat', marker='.', alpha=1, figsize=(16, 12), colormap='viridis'):
    df.plot.scatter(x=x, y=y, c=metric_col, title=metric_col
                    , edgecolors='none', colormap=colormap, marker=marker, alpha=alpha, figsize=figsize)
    plt.xticks([], [])
    plt.yticks([], [])


hex_col = 'hex' + str(aperture_size)

# find hexs containing the points
incident_df[hex_col] = incident_df.apply(lambda x: h3.geo_to_h3(x.lat, x.lng, aperture_size), 1)

# aggregate the points
incident_g_df = incident_df.groupby(hex_col).size().to_frame('cnt').reset_index()

# find center of hex for visualization
incident_g_df['lat'] = incident_g_df[hex_col].apply(lambda x: h3.h3_to_geo(x)[0])
incident_g_df['lng'] = incident_g_df[hex_col].apply(lambda x: h3.h3_to_geo(x)[1])

# pltot the hexs
plot_scatter(incident_g_df, metric_col='cnt', marker='o', figsize=(16, 12))
plt.title('hex-grid: accidents (incident)')

# In[426]:


grid_cell_hex_index = np.unique(incident_df[hex_col])

grid_cell_hex_bound = [h3.h3_to_geo_boundary(h) for h in grid_cell_hex_index]

grid_cells_hex = [Polygon([(x, y) for (y, x) in h]) for h in grid_cell_hex_bound]

grid_cell_index_by_id = dict((id(c), i) for i, c in zip(grid_cell_hex_index, grid_cells_hex))
grid_cell_index = STRtree(grid_cells_hex)

# In[427]:


likelihood = []

for waze_report_region in tqdm.tqdm(waze_report_uncertainty_regions):
    _probs = []
    _total_overlap_area = 0.0
    for grid_cell in grid_cell_index.query(waze_report_region):
        region_intersect = grid_cell.intersection(waze_report_region)
        _id = grid_cell_index_by_id[id(grid_cell)]
        _overlap_area = region_intersect.area
        _total_overlap_area += _overlap_area
        if _overlap_area > 0.0:
            _probs.append((_id, _overlap_area))
    _probs = [(i, p / _total_overlap_area) for i, p in _probs]
    likelihood.append(_probs)

# In[428]:


records = []
labels = []

label_aperture_size = 6
label_hex_col = 'hex' + str(label_aperture_size)

# Consider an accident occurring at a given time if values are in between this period
prev_dt = timedelta(minutes=10)  # consider pre accident/incident waze reports
post_dt = timedelta(minutes=5)  # consider post accident/incident waze reports

items = list(zip(report_df.time, report_df.reliability, likelihood))
for time, reliability, region_likelihood in tqdm.tqdm(items):
    record = {'time': time, 'reliability': reliability}
    record.update(dict({f'r_{r}': l for r, l in region_likelihood}))
    records.append(record)
    df_temp = incident_df[(incident_df.time >= (time - post_dt)) & (incident_df.time <= (time + prev_dt))]
    labels.append({h: 1 for h in df_temp[label_hex_col]})

records = pd.DataFrame(records).fillna(0)
labels = pd.DataFrame(labels).fillna(0)

records.shape, labels.shape

# In[437]:


labels.head()

# In[463]:


records.head()


# In[430]:


# P_I = None

# def p(r):
#     global P_I
#     if P_I is None:
#         ri, ci = records.reliability, (labels.sum(axis=1) > 0)
#         P_I = {ki: sum(ci[ri==ki]) / sum(ri==ki) for ki in set(ri)}
#     return P_I[r]

# p(7)
# P_I

def p(r):
    return r / 10


# In[502]:


def calculate_likelyhood(m):
    c = m.reliability.apply(p)
    m.drop('reliability', axis=1, inplace=True)
    if m.shape[0] == 0:
        c = 0
        m.loc[0] = np.zeros(m.shape[1])
    return (m.T * c).prod(axis=1).T


records[:5].set_index('time').resample('5T').apply(calculate_likelyhood)
