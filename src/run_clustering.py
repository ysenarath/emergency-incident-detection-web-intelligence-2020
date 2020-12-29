from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from model import h3_get_neighbors, calculate_posterior

df, features = None, None

for month in [10, 11, 12]:
    if df is None:
        df = pd.read_csv(f'output/waze/2019_{month}_raw.csv', index_col=0)
    else:
        df = df.append(pd.read_csv(f'output/waze/2019_{month}_raw.csv', index_col=0), sort=False)
    if features is None:
        features = pd.read_pickle(f'output/waze/2019_{month}_100_6_features.pkl')
    else:
        features = features.append(pd.read_pickle(f'output/waze/2019_{month}_100_6_features.pkl'), sort=False)

eps = 0.8

df = df.assign(relPubMillis=(df.pubMillis - np.min(df.pubMillis)) / 60000)
X = df[['lng', 'lat', 'relPubMillis']].values
clusterer = DBSCAN(eps=eps, min_samples=2).fit(X)
cluster_labels = []
c_max = np.max(clusterer.labels_)
for c in clusterer.labels_:
    if c == -1:
        c_max += 1
        cluster_labels.append(c_max)
    else:
        cluster_labels.append(c)
features = features.assign(cluster_label=cluster_labels)
df = df.assign(cluster_label=cluster_labels)


def predict_proba(df, incident_interval=25, time_step='5'):
    incident_interval = timedelta(minutes=incident_interval)
    features = []
    incident_id = 0
    regions = [r for r in df.columns if r.startswith('r_')]
    for label in tqdm(np.unique(df.cluster_label)):
        # for each region
        incident_records = df[df.cluster_label == label]
        col = regions[incident_records[regions].sum(axis=0).values.argmax()]
        print(col)
        col_sel = ['time', 'reliability', 'cluster_label'] + [col] + [x for x in h3_get_neighbors(col, regions) if
                                                                      x in df.columns]
        incident_time = np.min(incident_records.time)
        incident_records = incident_records.drop('cluster_label', axis=1)
        time_steps, posterior_probs = calculate_posterior(incident_records[col_sel], col, time_step)
        features_temp = []
        for time_step_val, posterior_proba in zip(time_steps, posterior_probs):
            if not np.math.isnan(posterior_proba):
                features_temp.append({
                    'incident_id': incident_id,
                    'start_time': incident_time,
                    'end_time': incident_time + incident_interval,
                    'time': time_step_val,
                    'region': col,
                    'posterior_proba': posterior_proba,
                })
        features += features_temp
        incident_id = incident_id + 1
    return pd.DataFrame(features)


posterior = predict_proba(features)
