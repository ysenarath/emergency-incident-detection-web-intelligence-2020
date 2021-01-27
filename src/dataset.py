import os

import h3
import pandas as pd

from config import config


def load_dataset(path=config['DEFAULT']['DATA_PATH']):
    """Loads dataset from the provided path.

    Arguments:
        path {string} -- path to dataset
    """
    return pd.read_csv(os.path.join(path, config['DEFAULT']['DATA_FILE']))


# Map labels with records
def label_mapper(features, incidents):
    mapped_labels = []
    for index, row in features.iterrows():
        label = incidents[
            (incidents.time >= row.start_time) & (incidents.time < row.end_time) & (incidents.region == row.region[2:])]
        mapped_labels.append(label.shape[0] > 0)
    return mapped_labels


def load_incidents(aperture_size=None, df_freq=None):
    if df_freq is None:
        df = pd.read_pickle('resources/tdot_testing_incidents.pk')
        df_freq = df.groupby(
            ['unit_segment_id', 'GPS Coordinate Latitude', 'GPS Coordinate Longitude',
             'timestamp']).size().reset_index()
        df_freq.set_axis(['segment_id', 'lat', 'lng', 'timestamp', 'count'], axis=1, inplace=True)
        df_freq.loc[:, 'segment_id'] = df_freq.segment_id.apply(int)
        df_freq.loc[:, 'time'] = pd.to_datetime(df_freq['timestamp'], unit='s')
    if aperture_size is not None:
        aperture_size = int(aperture_size)
        df_freq.loc[:, 'region'] = df_freq.apply(lambda x: h3.geo_to_h3(x.lat, x.lng, aperture_size), 1)
    return df_freq
