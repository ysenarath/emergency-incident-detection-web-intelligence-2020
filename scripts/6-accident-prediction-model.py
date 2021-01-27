#!/usr/bin/env python
# coding: utf-8

# In[18]:
from datetime import timedelta

import numpy as np
import pandas as pd
import tqdm

# In[125]:

records = pd.read_pickle('../output/waze_12_features.pkl')
labels = pd.read_pickle('../output/waze_12_labels.pkl')

# In[249]:


labels.loc[:, 'region'] = labels.hex6.apply(lambda x: 'r_' + x)

# ### Incident Time Intervel
# 
# - Definition: The average time taken for the arrival of first waze repot to the last waze report

# In[250]:


labels.head()

# In[179]:


# calculate the time difference between two consecative incidents in same region
consec_inc_td = []

for region in labels.hex6.unique():
    time = labels[labels.hex6 == region].sort_values('time').time.diff().dropna().astype('timedelta64[m]')
    consec_inc_td += time[time.notnull()].tolist()

consec_inc_td_df = pd.Series(consec_inc_td).sort_values()

consec_inc_td_df[consec_inc_td_df < 120].plot.hist(bins=15, alpha=0.8)

# In[228]:


# T' calculation

number_of_reports = []

pre_dt = timedelta(minutes=60)

for region in labels.hex6.unique():
    region_df = labels[labels.hex6 == region]
    col = 'r_' + region
    if col in records:
        records_after = records.loc[records[col] > 0, ['time', 'reliability', col]]
        for incident_time in region_df.time.sort_values():
            records_before = records_after[
                ((incident_time - pre_dt) <= records_after.time) & (records_after.time <= incident_time)]
            time_intervels = (incident_time - records_before.sort_values('time').time).astype('timedelta64[m]')
            number_of_reports += time_intervels.tolist()
            records_after = records_after[records_after.time > incident_time]

# In[367]:


number_of_reports_df = pd.Series(number_of_reports).sort_values()

number_of_reports_df[number_of_reports_df < 24].plot.hist(bins=24, alpha=0.8, label='Time')


# ### Estimate posterior probabilities every 5 min (keep doing this up to T' min)

# In[421]:


# change to historical data values
# P(I=1,R1)
def region_prior(region):
    # use labels_prv to prevent overfitting
    return sum(labels.region == region) / labels.shape[0]


# In[25]:


def p(r):
    return r / 10


# In[426]:


def calculate_likelihood(m):
    c = m.reliability.apply(p)
    m.drop('reliability', axis=1, inplace=True)
    if m.shape[0] == 0:
        c = 0
        m.loc[0] = np.zeros(m.shape[1])
    return (m.T * c).T.prod(axis=0)


# In[457]:


def calculate_posterior(df, r, time_step=5):
    time_steps_records = df.set_index('time').resample(f'{time_step}T')
    likelihood = time_steps_records.apply(calculate_likelihood)
    likelihood = likelihood[likelihood[r] > 0]
    region_priors = []
    for c in likelihood:
        region_priors.append(region_prior(c))
    region_priors = np.array(region_priors)
    likelihood.iloc[0, :] = region_priors * likelihood.iloc[0, :]
    p_0 = (1 - np.sum(region_priors))
    l_0 = (1 - likelihood).prod(axis=1)
    p_n = likelihood.cumprod(axis=0)
    p_d = np.sum(p_n, axis=1) + p_0 * l_0
    posterior = p_n[r] / p_d
    return likelihood.index, posterior


# In[458]:

def incident_posterior(df, incident_interval=25):
    incident_interval = timedelta(minutes=incident_interval)
    features = []
    incident_id = 0
    for col in tqdm.tqdm(df.columns):
        # for each region
        if col.startswith('r_'):
            df_region = df.loc[df[col] > 0, :]
            incident_time = np.min(df_region.time)
            while incident_time <= np.max(df_region.time):
                incident_records = df_region[
                    (incident_time <= df_region.time) & (df_region.time < (incident_time + incident_interval))]
                time_steps, posterior_probs = calculate_posterior(incident_records, col)
                features_temp = []
                for time_step, posterior_proba in zip(time_steps, posterior_probs):
                    features_temp.append({
                        'incident_id': incident_id,
                        'start_time': incident_time,
                        'end_time': incident_time + incident_interval,
                        'time': time_step,
                        'region': col,
                        'posterior_proba': posterior_proba,
                    })
                features += features_temp
                incident_time = np.min(df_region[(incident_time + incident_interval) <= df_region.time].time)
                incident_id = incident_id + 1
    return pd.DataFrame(features)


features = incident_posterior(records)

features.head()

# In[459]:


# number of incidents with the number of waze reports

pd.DataFrame(np.unique(features.groupby('incident_id').size(), return_counts=True))


# # Incident Prediction

# In[461]:


# Map labels with records
def label_mapper(features, labels):
    mapped_labels = []
    for index, row in features.iterrows():
        label = labels[(labels.time >= row.start_time) & (labels.time < row.end_time) & (labels.region == row.region)]
        mapped_labels.append(label.shape[0] > 0)
    return mapped_labels


# In[462]:


features.loc[:, 'label'] = label_mapper(features, labels)

# In[463]:


features.head()

# In[464]:


np.unique(features.label, return_counts=True)

# In[465]:


features.loc[:, 'hour'] = features.time.dt.hour

features.loc[:, 'is_peak'] = ((1 <= features.hour) & (features.hour <= 4)) | (
        (10 <= features.hour) & (features.hour <= 14))

features.head()

# In[466]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

feature_extractor = ColumnTransformer([
    ('posterior_proba', 'passthrough', ['posterior_proba']),
    # ('is_peak', 'passthrough', ['is_peak']),
    ('hour_enc', OneHotEncoder(), ['hour']),
    ('region_enc', OneHotEncoder(), ['region']),
])

X = feature_extractor.fit_transform(features)
y = features.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

reg = LogisticRegression(random_state=42, class_weight='balanced').fit(X_train, y_train)

y_pred = reg.predict(X_test)

reg.score(X_test, y_test)

# In[469]:

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

# In[ ]:
