# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import datetime

import numpy as np
import pandas as pd
import streamlit as st

st.title('WAZE-TDOT Data Exploration')

incident_df = pd.read_pickle('output/_waze_temp/tdot_12_incidents.pkl')
incident_df['time'] = pd.to_datetime(incident_df['timestamp'], unit='s')
st.write('Accident Incident Data: ')
st.write(incident_df)

report_df = pd.read_csv('output/_waze_temp/waze_12_etrims.csv', index_col=0)
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

date_to_filter = st.date_input('Date to Filter')  # min: 0h, max: 23h, default: 17h
time_to_filter = datetime.datetime.combine(date_to_filter,
                                           st.time_input('Time to Filter'))  # min: 0h, max: 23h, default: 17h

period = datetime.timedelta(minutes=st.slider('Period', 0, 60, 25))

filtered_report_df = report_df[
    (report_df.time >= time_to_filter - period) & (report_df.time <= time_to_filter + period)]
report_map_data = filtered_report_df.loc[:, ['lat', 'lng']].copy()
report_map_data.columns = ['lat', 'lon']

filtered_incident_df = incident_df[
    (incident_df.time >= time_to_filter - period) & (incident_df.time <= time_to_filter + period)]
incident_map_data = filtered_incident_df.loc[:, ['lat', 'lng']].copy()
incident_map_data.columns = ['lat', 'lon']

st.write(report_map_data)
st.write(filtered_incident_df)

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
