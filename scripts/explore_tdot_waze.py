import math
from datetime import timedelta

# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import streamlit as st
from h3 import geo_to_h3

st.title('WAZE-TDOT Data Exploration')

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

#
# ------------------- Analise Data by Hour of Day --------------------------------------------------------------------
#

st.subheader('Analysis of Data by Hour of Day')

report_hist_values = np.histogram(report_df.time.dt.hour, bins=24, range=(0, 24))[0]
incident_hist_values = np.histogram(incident_df.time.dt.hour, bins=24, range=(0, 24))[0]

st.bar_chart(pd.DataFrame({
    'Reports': report_hist_values,
    'Accidents': incident_hist_values,
}))

# Maps by Hour of Day

hour_to_filter = st.slider('hour', 0, 23, 8)  # min: 0h, max: 23h, default: 17h
period = st.slider('period', 0, 23, 1)  # min: 0h, max: 23h, default: 17h

filtered_report_df = report_df[
    (report_df.time.dt.hour >= hour_to_filter - period) & (report_df.time.dt.hour <= hour_to_filter + period)]
report_map_data = filtered_report_df.loc[:, ['lat', 'lng']].copy()
report_map_data.columns = ['lat', 'lon']

filtered_incident_df = incident_df[
    (incident_df.time.dt.hour >= hour_to_filter - period) & (incident_df.time.dt.hour <= hour_to_filter + period)]
incident_map_data = filtered_incident_df.loc[:, ['lat', 'lng']].copy()
incident_map_data.columns = ['lat', 'lon']

st.write('Map of all Reports and Accidents')

viewport = {
    "latitude": np.average([np.max(report_map_data.lat), np.min(report_map_data.lat)]),
    "longitude": np.average([np.max(report_map_data.lon), np.min(report_map_data.lon)]),
    'zoom': 10
}
layers = [
    {"data": report_map_data, "type": "ScatterplotLayer", 'getFillColor': [0, 0, 255]},
    {"data": incident_map_data, "type": "ScatterplotLayer", 'getFillColor': [255, 0, 0]}
]

st.deck_gl_chart(viewport=viewport, layers=layers)

# Reports related to same incident based on time

st.write('Reporting rate for Incidents by Time Segments')

max_dist = st.slider('Grid Resolution', 0, 15, 10)

incident_df['h3_grid'] = [geo_to_h3(lat, lng, max_dist) for lat, lng in zip(incident_df.lat, incident_df.lng)]
report_df['h3_grid'] = [geo_to_h3(lat, lng, max_dist) for lat, lng in zip(report_df.lat, report_df.lng)]

prev_td = (5, 10)
post_td = (3, 5)

st.write(f'No of Grids: {len(np.unique(report_df.h3_grid))}')

avg_count_post_t = []
column_names = []
progress_bar = st.progress(0)
progress_bar_len = (post_td[1] + 1 - post_td[0]) * (prev_td[1] + 1 - prev_td[0])
progress_bar_count = 0.0
for ix, post_t in enumerate(range(post_td[0], post_td[1] + 1)):
    column_names.append(post_t)
    avg_count_prev_t = []
    post_t = timedelta(minutes=post_t)
    for prev_t in range(prev_td[0], prev_td[1] + 1):
        counts = []
        prev_t = timedelta(minutes=prev_t)
        for i in incident_df.to_dict(orient="row"):
            t = i['time']
            filtered_report_df = report_df[(report_df.time >= (t - prev_t)) & (report_df.time <= (t + post_t))]
            filtered_report_in_grid_df = filtered_report_df[filtered_report_df.h3_grid == i['h3_grid']]
            counts.append(len(filtered_report_in_grid_df))
        progress_bar_count += 1
        progress_bar.progress(math.floor(progress_bar_count * 100 / progress_bar_len))
        avg_count_prev_t.append(np.average(counts))
    avg_count_post_t.append(avg_count_prev_t)
avg_count_post_t = np.array(avg_count_post_t).transpose()

chart_data = pd.DataFrame(avg_count_post_t, columns=column_names)

st.line_chart(chart_data)
