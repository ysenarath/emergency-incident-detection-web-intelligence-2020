from tap.dataset import load_dataset

if __name__ == '__main__':
    df = load_dataset()
    df['date'] = ['{}_{}_{}'.format(r.month, r.day, r.year) for r in df.date_time]
    df['hod'] = ['{}:{}'.format(r.hour, ['00', 30][r.minute < 30]) for r in df.date_time]

    summary_df = df.groupby(['grid', 'date', 'hod']).size().to_frame('freq').reset_index()

    print(summary_df.head())
    # summary_df.to_csv('output/waze_spatiotemporal_freq.csv', header=True, index=False)
    # summary_df = summary_df[['hod', 'freq']].groupby(['hod']).agg(['count', 'sum', 'mean'])
    # summary_df.columns = summary_df.columns.droplevel()
    # summary_df = summary_df.reset_index()
    # print(summary_df.head())
    # summary_df.to_csv('output/waze_temporal_freq.csv', header=True, index=False)
    #
    # count_df = df.groupby(['grid', 'hod']).size().to_frame('freq')
    # print(count_df)
    # count_df.to_csv('output/waze_spatiotemporal_freq.csv', header=True)
    #
    # count_df = df.groupby('hod').size().to_frame('freq')
    # count_df.reset_index(level=0, inplace=True)
    # print(count_df)
    # count_df.to_csv('output/waze_temporal_freq.csv', header=True)
