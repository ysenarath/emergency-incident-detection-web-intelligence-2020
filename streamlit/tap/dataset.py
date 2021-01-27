import datetime
import os
import time

import numpy as np
import pandas as pd

from tap.grids import get_grids

__all__ = [
    'to_datetime',
    'load_dataset',
]


def to_timestamp(df):
    month, day, year = df[0].split(' ')[0].split('/')
    mins = int(int(df[1]) % 100)
    hrs = int((int(df[1]) - mins) / 100)
    return datetime.datetime(int(year), int(month), int(day), hrs, mins).timestamp() * 1000


def to_datetime(ts):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts / 1000))


def format_time(x):
    try:
        st = x.split('T')[0] + ' ' + x.split('T')[1].split('.')[0]
    except IndexError:
        return None
    return st


def preprocess_incidents():
    df = pd.read_csv('data/fd_export_only_incidents.csv')
    df = df.mask(df.eq('None')).dropna()
    df = df.loc[:, ['incidentNumber', 'latitude', 'longitude', 'alarmDateTime']]
    df = df.rename(columns={'latitude': 'x', 'longitude': 'y', 'alarmDateTime': 'date_time'})
    df['date_time'] = pd.to_datetime(df.date_time.apply(format_time))
    df['pubMillis'] = df['date_time'].astype(np.int64) / int(1e6)
    df['grid'] = get_grids(zip(df.x, df.y))
    original_size = len(df)
    df = df.drop_duplicates()
    processed_size = len(df)
    print(df.head())
    print('Dropped %i out of %i' % (original_size - processed_size, original_size))
    df.to_csv('data/nfd_data_all_precessed.csv', index=False)


def load_dataset(option=None):
    if option is None or option.lower() == 'waze':
        df = pd.read_csv('data/accident_data_all_precessed.csv')
        df['date_time'] = pd.to_datetime(df.date_time)
    elif option.lower() == 'nfd':
        if not os.path.exists('data/nfd_data_all_precessed.csv'):
            preprocess_incidents()
        df = pd.read_csv('data/nfd_data_all_precessed.csv')
    else:
        df = pd.read_csv('data/march_april.csv')
        df['pubMillis'] = df[['Date of Crash', 'Time of Crash']].apply(to_timestamp, axis=1)
        df['date_time'] = pd.to_datetime(list(map(to_datetime, df.pubMillis)))
        df = df.loc[:, ['GPS Coordinate Latitude', 'GPS Coordinate Longitude', 'date_time', 'pubMillis', 'grid']]
        df = df.rename(columns={'GPS Coordinate Longitude': 'y', 'GPS Coordinate Latitude': 'x'})
    return df
