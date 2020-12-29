from datetime import timedelta

import h3
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate as _cross_validate, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from dataset import load_incidents

df_freq = load_incidents()


def h3_get_neighbors(region, columns=None):
    if region.startswith('r_'):
        region = region[2:]
    return [x for x in np.unique(columns) if h3.h3_indexes_are_neighbors(region, x[2:])]


def region_prior_2(region, time):
    global df_freq
    df_freq = df_freq[df_freq.time < time]
    resolution = h3.h3_get_resolution(region[2:])
    incident_df = load_incidents(resolution, df_freq)
    lower = len(incident_df) + len(np.unique(incident_df.region)) * 24
    prior = (np.sum((incident_df.region == region[2:]) & (incident_df.time.dt.hour == time.hour)) + 1) / lower
    return prior


def region_prior(region, time):
    global df_freq
    df_freq = df_freq[df_freq.time < '2019-10-01']
    resolution = h3.h3_get_resolution(region[2:])
    incident_df = load_incidents(resolution, df_freq)
    # change to historical data values
    # use labels_prv to prevent overfitting
    # return 1 / len(np.unique(labels.region))
    lower = len(incident_df) + len(np.unique(incident_df.region)) * 24
    prior = (np.sum((incident_df.region == region[2:]) & (incident_df.time.dt.hour == time.hour)) + 1) / lower
    # prior = np.sum(incident_df['region'] == region[2:]) / len(incident_df)
    return prior


def p(r):
    # WAZE report reliability probability
    return r / 10


def calculate_posterior(df, r, time_step):
    df = df.assign(reliability_1=df.loc[:, 'reliability'].apply(p))
    df = df.assign(reliability_0=1 - df.loc[:, 'reliability_1'])
    df = df.drop('reliability', axis=1)
    sec = ''
    if time_step.endswith('.5'):
        time_step = time_step.split('.')[0]
        sec = '30S'
    matrix = df.set_index('time').resample(f'{time_step}T{sec}').apply(np.prod)
    p_w_1 = matrix.loc[:, 'reliability_1']
    matrix = matrix.drop('reliability_1', axis=1)
    p_w_0 = matrix.loc[:, 'reliability_0']
    matrix = matrix.drop('reliability_0', axis=1)
    # Calculate region priors
    prior = np.array([region_prior(c, matrix.index[0]) for c in matrix])
    p_1 = np.sum(prior)
    p_0 = 1 - p_1
    p_1_w = p_w_1.iloc[0] * p_1 / (p_w_1.iloc[0] * p_1 + p_w_0.iloc[0] * p_0)
    matrix.iloc[0, :] = p_1_w * (matrix.iloc[0, :] * prior / np.sum(matrix.iloc[0, :] * prior))
    for ridx in range(1, matrix.shape[0]):
        prior = matrix.iloc[ridx - 1, :]
        p_1, p_0 = p_1_w, 1 - p_1_w
        p_1_w = p_w_1.iloc[ridx] * p_1 / (p_w_1.iloc[ridx] * p_1 + p_w_0.iloc[ridx] * p_0)
        matrix.iloc[ridx, :] = p_1_w * (matrix.iloc[ridx, :] * prior / np.sum(matrix.iloc[ridx, :] * prior))
    # Return posteriors
    return matrix.index, matrix.loc[:, r]


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


def extract_features(x, df):
    avg_reliabilities = []
    counts = []
    priors = []
    for i in tqdm(range(x.shape[0])):
        row = x.iloc[i]
        s, e = row.start_time, row.end_time
        temp = df[(df.time >= s) & (df.time <= e)]
        priors.append(region_prior_2(row.region, s))
        avg_reliabilities.append(np.average(temp.reliability))
        counts.append(len(temp))
    x = x.assign(avg_reliability=avg_reliabilities, count=counts, rand_porba=np.random.rand(x.shape[0]), priors=priors)
    return x


def cross_validate(x, y, model='LogisticRegression'):
    feature_extractor_1 = ColumnTransformer([
        ('posterior_proba', 'passthrough', ['posterior_proba']),
        # ('priors', 'passthrough', ['priors']),
        # ('rand_porba', 'passthrough', ['rand_porba']),
        # ('is_peak', 'passthrough', ['is_peak']),
        # ('hour_enc', OneHotEncoder(), ['hour']),
        # ('region_enc', OneHotEncoder(), ['region']),
    ])
    feature_extractor = ColumnTransformer([
        ('avg_reliability', 'passthrough', ['avg_reliability']),
        ('count', 'passthrough', ['count']),
    ])
    # X = feature_extractor.fit_transform(x)
    if model == 'LogisticRegression':
        reg1 = LogisticRegression(random_state=42, class_weight='balanced')
        reg2 = LogisticRegression(random_state=42, class_weight='balanced')
    elif model == 'DecisionTreeClassifier':
        reg1 = DecisionTreeClassifier(max_depth=5, class_weight='balanced')
        reg2 = DecisionTreeClassifier(max_depth=5, class_weight='balanced')
    elif model == 'RandomForestClassifier':
        reg1 = RandomForestClassifier(max_depth=3, class_weight='balanced')
        reg2 = RandomForestClassifier(max_depth=3, class_weight='balanced')
    if model == 'LogisticRegression':
        reg = LogisticRegression(random_state=42, class_weight='balanced')
    elif model == 'DecisionTreeClassifier':
        reg = DecisionTreeClassifier(max_depth=5, class_weight='balanced')
    elif model == 'RandomForestClassifier':
        reg = RandomForestClassifier(max_depth=3, class_weight='balanced')
    pipe = StackingClassifier(estimators=[('reg2', make_pipeline(feature_extractor_1, reg2)),
                                          ('reg1', make_pipeline(feature_extractor, reg1))],
                              final_estimator=reg)
    cv_results = _cross_validate(pipe, x, y, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'], cv=5)
    y_pred = cross_val_predict(pipe, x, y, cv=5)
    return cv_results, y_pred
