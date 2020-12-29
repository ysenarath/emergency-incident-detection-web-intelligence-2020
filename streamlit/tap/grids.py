import json

import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from pyproj import Proj

from tap.dataset import to_datetime

_project = Proj(
    '+proj=lcc +lat_1=36.41666666666666 +lat_2=35.25 +lat_0=34.33333333333334 +lon_0=-86 '
    '+x_0=600000 +y_0=0 +ellps=GRS80 +datum=NAD83 +no_defs'
)


def read_grids():
    grid = []
    for grid_h in np.load('data/grids.npy', encoding='bytes', allow_pickle=True):
        h_points = []
        for grid_y in grid_h:
            y_points = []
            for point in grid_y:
                long, lat = _project(point[0], point[1], inverse=True)
                y_points.append([long, lat])
            h_points.append(y_points)
        grid.append(h_points)
    return grid


def get_grids(coordinates):
    grids = read_grids()
    grid_values = []
    for dx, dy in coordinates:
        grid = None
        for y in range(len(grids)):
            for x in range(len(grids[y])):
                assert grids[y][x][0][0] < grids[y][x][1][0] and grids[y][x][0][1] < grids[y][x][1][1], 'Invalid Grid'
                if grids[y][x][0][0] <= dy < grids[y][x][1][0] and grids[y][x][0][1] <= dx < grids[y][x][1][1]:
                    grid = (y, x)
                    break
            if grid is not None:
                break
        grid_values.append(grid)
    return grid_values


if __name__ == '__main__':
    df = json_normalize(json.load(open('data/accident_data_all.json')))
    df['date_time'] = pd.to_datetime(list(map(to_datetime, df.pubMillis)))
    # df['Month'] = df['date_time'].apply(lambda t: t.month)
    # df['Day'] = df['date_time'].apply(lambda t: t.day)
    # df['Year'] = df['date_time'].apply(lambda t: t.year)
    # df['Time'] = df['date_time'].apply(lambda t: '%02i%02i' % (t.hour, t.minute))
    # df['Seconds'] = df['date_time'].apply(lambda t: t.second)
    df = df[['location.x', 'location.y', 'date_time', 'pubMillis', 'reliability', 'magvar']]
    df = df.rename(columns={'location.x': 'y', 'location.y': 'x'})
    df['grid'] = get_grids(zip(df.x, df.y))
    original_size = len(df)
    df = df.drop_duplicates()
    processed_size = len(df)
    print('Dropped %i out of %i' % (original_size - processed_size, original_size))
    df.to_csv('data/accident_data_all_precessed_extra.csv', index=False)
