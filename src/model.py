from datetime import timedelta

import h3
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate as _cross_validate
from tqdm import tqdm

from dataset import load_incidents

df_freq = load_incidents()


def h3_get_neighbors(region, columns=None):
    if region.startswith('r_'):
        region = region[2:]
    return [x for x in np.unique(columns) if h3.h3_indexes_are_neighbors(region, x[2:])]


def region_prior(region, time):
    resolution = h3.h3_get_resolution(region[2:])
    incident_df = load_incidents(resolution, df_freq)
    # change to historical data values
    # use labels_prv to prevent overfitting
    # return 1 / len(np.unique(labels.region))
    # return np.sum((incident_df.region == region[2:]) & (time.hour == incident_df.time.dt.hour)) / len(incident_df)
    return np.sum(incident_df.region == region[2:]) / len(incident_df)


def p(r):
    # WAZE report reliability probability
    return r / 10


def calculate_likelihood(m):
    c = m.reliability.apply(p)
    m.drop('reliability', axis=1, inplace=True)
    if m.shape[0] == 0:
        m.loc[0], c = np.zeros(m.shape[1]), 0
    return (m.T * c).T.prod(axis=0)


def calculate_posterior(df, r, time_step):
    sec = ''
    if time_step.endswith('.5'):
        time_step = time_step.split('.')[0]
        sec = '30S'
    time_steps_records = df.set_index('time').resample(f'{time_step}T{sec}')
    # Matrix - containes total likelihood
    matrix = time_steps_records.apply(calculate_likelihood)
    # Find likelihood, prior for first time period
    likelihood = matrix.iloc[0, :]
    prior = np.array([region_prior(c, matrix.index[0]) for c in matrix])
    # Calcualate prior for I = 0 for first time period
    p_0 = (1 - np.sum(prior))
    # Calcualate likelihood for I = 0 for all periods using reliability of waze reports prod(1-p)
    l_0 = time_steps_records.apply(lambda m: (1 - m.reliability.apply(p)).prod())
    denom = np.sum(likelihood * prior) + l_0[0] * p_0
    # For each time step calculate posterior (start with zero row - special case)
    # matrix - containes posteriors [partly]
    matrix.iloc[0, :] = likelihood * prior / denom
    p_0 = l_0[0] * p_0 / denom
    return_row = [True for _ in range(matrix.shape[0])]
    for ridx in range(1, matrix.shape[0]):
        prior = matrix.iloc[ridx - 1, :]
        likelihood = matrix.iloc[ridx, :]
        if np.sum(likelihood) == 0:
            matrix.iloc[ridx, :] = prior
            return_row[ridx] = False
            continue
        # Calculate denominator of posterior
        denom = np.sum(likelihood * prior) + l_0[ridx] * p_0
        # Calculate posterior
        matrix.iloc[ridx, :] = likelihood * prior / denom
        p_0 = l_0[ridx] * p_0 / denom
    # Return posteriors
    return matrix[return_row].index, matrix[return_row].loc[:, r]


def predict_proba(df, incident_interval=25, time_step='5'):
    incident_interval = timedelta(minutes=incident_interval)
    features = []
    incident_id = 0
    regions = [r for r in df.columns if r.startswith('r_')]
    for col in tqdm(regions):
        # for each region
        col_sel = ['time', 'reliability', col] + [x for x in h3_get_neighbors(col, regions) if x in df.columns]
        df_region = df.loc[df[col] > 0, col_sel]
        incident_time = np.min(df_region.time)
        while incident_time <= np.max(df_region.time):
            incident_records = df_region[
                (incident_time <= df_region.time) & (df_region.time < (incident_time + incident_interval))]
            time_steps, posterior_probs = calculate_posterior(incident_records, col, time_step)
            features_temp = []
            for time_step_val, posterior_proba in zip(time_steps, posterior_probs):
                features_temp.append({
                    'incident_id': incident_id,
                    'start_time': incident_time,
                    'end_time': incident_time + incident_interval,
                    'time': time_step_val,
                    'region': col,
                    'posterior_proba': posterior_proba,
                })
            features += features_temp
            incident_time = np.min(df_region[(incident_time + incident_interval) <= df_region.time].time)
            incident_id = incident_id + 1
    return pd.DataFrame(features)


def cross_validate(x, y):
    feature_extractor = ColumnTransformer([
        ('posterior_proba', 'passthrough', ['posterior_proba']),
        # ('is_peak', 'passthrough', ['is_peak']),
        # ('hour_enc', OneHotEncoder(), ['hour']),
        # ('region_enc', OneHotEncoder(), ['region']),
    ])
    X = feature_extractor.fit_transform(x)
    reg = LogisticRegression(random_state=42, class_weight='balanced')
    cv_results = _cross_validate(reg, X, y, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'], cv=5)
    return cv_results
