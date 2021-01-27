import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

from tap.dataset import load_dataset, to_datetime


def main():
    st.title('Disaster Relief Dataset Analysis')
    st.header('Instructions')
    st.write(
        'Use this platform to interact with WAZE and T.DOT Datasets. '
        'Only data for the duration feb-march 2019 is updated.'
        'All available accident related WAZE data is added.'
    )
    st.header('Temporal Analysis')
    option = st.selectbox('Select Dataset: ', ('WAZE', 'TDOT', 'NFD'))
    df = load_dataset(option)
    st.write('Dataset Summary: ')
    st.write(df.describe())
    start_time = st.slider('Select start time (T_MID):', min(df.pubMillis), max(df.pubMillis))
    period = st.slider('Select time period (Minutes):', 0, 60, 30) * 30000
    st.write(
        'Selected Duration: %s to %s ' % (
            to_datetime(start_time - period),
            to_datetime(start_time + period)
        )
    )
    data = df[df.pubMillis.between(start_time - period, start_time + period)]
    map_data = pd.DataFrame(data[['x', 'y']].values, columns=['lat', 'lon'])
    st.subheader('Selected Data Sample: ')
    st.write(map_data)
    st.subheader('World Map: ')
    st.map(map_data)
    st.subheader('HeatMap: ')
    # Convert this grid to columnar data expected by Altair
    print(df.columns)
    source = df.copy()
    source = source[df.pubMillis.between(1550262948764, 1564679814000)]
    source.loc[:, 'z'] = 1
    source = source.groupby('grid').count()
    source['x'] = source.index.to_series().apply(lambda x: eval(x)[0])
    source['y'] = source.index.to_series().apply(lambda y: eval(y)[1])
    source = source[['x', 'y', 'z']]
    st.write(source)
    chart = alt.Chart(source).mark_rect().encode(
        y='y:O',
        x='x:O',
        color='z:Q'
    )
    st.altair_chart(chart)
    # HOD Size
    st.write('Spatiotemporal')
    df['date_time'] = pd.to_datetime(df.date_time)
    df['date'] = ['{}_{}_{}'.format(r.month, r.day, r.year) for r in df.date_time]
    df['hod'] = ['{:0>2d}:{}'.format(r.hour, ['00', 30][r.minute < 30]) for r in df.date_time]

    summary_df = df.groupby(['grid', 'date', 'hod']).size().to_frame('freq').reset_index()
    summary_df = summary_df[['hod', 'freq']].groupby(['hod']).agg(['count', 'mean'])
    summary_df.columns = summary_df.columns.droplevel()
    summary_df = summary_df.reset_index()

    st.write(summary_df)
    chart = alt.Chart(pd.DataFrame(
        {'Time Interval Start (30 mins)': summary_df['hod'],
         'Average Incident Freq': summary_df['mean']})).mark_line().encode(
        y='Average Incident Freq',
        x='Time Interval Start (30 mins)',
    )
    st.altair_chart(chart)


if __name__ == '__main__':
    main()
