"""

Calculation Steps

1. python src/run.py preprocess --year 2019
2. python src/run.py extract-uncertainty-regions --year 2019 --month 12 --radius 1000
3. python src/run.py extract-features --year 2019 --month 12 --radius 1500 --aperture_size 6
4. python src/run.py predict --year 2019 --month 12 --radius 500 --aperture_size 6  --incident_interval 25 --time_step 5

"""

import codecs
import json
import os

import click
import geopandas as gp
import h3
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from tqdm import tqdm

import model
from dataset import load_incidents, label_mapper
from utils import project_coordinates


@click.group()
@click.option('--debug/--no-debug', default=False)
def cli(debug):
    click.echo('Debug mode is %s' % ('on' if debug else 'off'))


@cli.command()
@click.option('--year', default=2019, help='Year to preprocess.')
def preprocess(year):
    for month in os.listdir(f'data/waze/{year}'):
        if not os.path.exists(f'output/waze/{year}_{month}_raw.csv'):
            data = []
            errors = []
            pbar = tqdm(os.listdir(f'data/waze/{year}/{month}'))
            pbar.set_description("Processing month: %s" % month)
            for day in pbar:
                for hr in os.listdir(f'data/waze/{year}/{month}/' + day):
                    try:
                        var = json.load(codecs.open(f'data/waze/{year}/{month}/{day}/{hr}', 'r', 'utf-8-sig'))
                        data.append(var)
                    except:
                        errors.append({'day': day, 'hr': hr})
            accidents = []
            for alerts in tqdm(data):
                if 'alerts' in alerts:
                    for alert in alerts['alerts']:
                        if alert['type'] == 'ACCIDENT':
                            alert['lng'] = alert['location']['x']
                            alert['lat'] = alert['location']['y']
                            accidents += [alert]
            accident_df = pd.DataFrame.from_dict(accidents)
            accident_df.drop('location', axis=1, inplace=True)
            accident_df = accident_df.drop_duplicates()
            accident_df.to_csv(f'output/waze/{year}_{month}_raw.csv')


@cli.command()
@click.option('--year', default=2019, help='Year to preprocess.')
@click.option('--month', default=None, help='Month to preprocess.')
@click.option('--radius', default=None, help='Radius of WAZE report effect (in \'m\').')
# 500, 1000, 1500
def extract_uncertainty_regions(year, month, radius):
    df = pd.read_csv(f'output/waze/{year}_{month}_raw.csv', index_col=0)
    gdf = gp.GeoDataFrame(df, geometry=gp.points_from_xy(df.lng, df.lat))
    uncertainty_regions = []
    for lng, lat in tqdm(list(zip(df.lng, df.lat))):
        uncertainty_regions.append(project_coordinates(lng, lat, radius))
    gdf.loc[:, 'region'] = uncertainty_regions
    gdf.to_pickle(f'output/waze/{year}_{month}_{radius}_region.pkl')


@cli.command()
@click.option('--year', default=2019, help='Year to preprocess.')
@click.option('--month', default=None, help='Month to preprocess.')
@click.option('--radius', default=None, help='Radius of WAZE report effect (in \'m\').')
@click.option('--aperture_size', default=None, help='H3 grid resolution.')
# 6 and 7
def extract_features(year, month, radius, aperture_size):
    incident_df = load_incidents(aperture_size)
    grid_cell_hex_index = np.unique(incident_df.loc[:, 'region'])
    grid_cell_hex_bound = [h3.h3_to_geo_boundary(h) for h in grid_cell_hex_index]
    grid_cells_hex = [Polygon([(x, y) for (y, x) in h]) for h in grid_cell_hex_bound]
    grid_cell_index_by_id = dict((id(c), i) for i, c in zip(grid_cell_hex_index, grid_cells_hex))
    grid_cell_index = STRtree(grid_cells_hex)

    gdf = pd.read_pickle(f'output/waze/{year}_{month}_{radius}_region.pkl')
    gdf['time'] = pd.to_datetime(gdf['pubMillis'], unit='ms')
    waze_report_uncertainty_regions = gdf.loc[:, 'region']
    likelihood = []
    for waze_report_region in tqdm(waze_report_uncertainty_regions):
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

    records = []
    # Consider an accident occurring at a given time if values are in between this period
    items = list(zip(gdf.loc[:, 'time'], gdf.loc[:, 'reliability'], likelihood))
    for time, reliability, region_likelihood in tqdm(items):
        record = {'time': time, 'reliability': reliability}
        record.update(dict({f'r_{r}': l for r, l in region_likelihood}))
        records.append(record)
        # df_temp = incident_df[(incident_df.time >= (time - post_dt)) & (incident_df.time <= (time + prev_dt))]
    records = pd.DataFrame(records).fillna(0)
    records.to_pickle(f'output/waze/{year}_{month}_{radius}_{aperture_size}_features.pkl')


@cli.command()
@click.option('--year', default=2019, help='Year to preprocess.')
@click.option('--month', default=None, help='Month to preprocess.')
@click.option('--radius', default=None, help='Radius of WAZE report effect (in \'m\').')
@click.option('--aperture_size', default=None, help='H3 Grid Resolutions.')
@click.option('--incident_interval', default=None, help='Incident interval.')
@click.option('--time_step', default=None, help='Time step period.')
@click.option('--select', default=None, help='Select part of the data for experiment.')
def predict(year, month, radius, aperture_size, incident_interval, time_step, select):
    config = {'year': year, 'month': month, 'radius': radius, 'aperture_size': aperture_size,
              'incident_interval': incident_interval, 'time_step': time_step, 'select': select}
    df = pd.read_pickle(f'output/waze/{year}_{month}_{radius}_{aperture_size}_features.pkl')
    if select is not None:
        df = df[-1 * int(select):]
    x = model.predict_proba(df, int(incident_interval), time_step)
    incident_df = load_incidents(aperture_size)
    y = label_mapper(x, incident_df)
    print(np.unique(y, return_counts=True))
    cv_results = model.cross_validate(x, y)
    for k, v in config.items():
        print(k, v)
    for k, v in cv_results.items():
        print(k, np.average(v))


# Test Parameters
# -----------------------------------
# WAZE Radius - 500, 1000, 1500
# H3 Grid Resolutions - 6 and 7
# T' - incident interval -- 5, 15, 25
# T - time step -- 1, 2.5, 5

# priors - sept

if __name__ == '__main__':
    cli()
