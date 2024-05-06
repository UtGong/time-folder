import pandas as pd

def data_transformation(data_name, time_column, stop_column, classification_column):
    data_path = 'data/' + data_name + '.csv'
    data = pd.read_csv(data_path)
    # filter dataset to only keep the first row of each same (TRIP ID,TRAIN ID,STOP ID), then save it to 'data/case3.csv'
    data = data.drop_duplicates(subset=["TRIP ID","TRAIN ID","STOP ID"], keep="first")
    # make 'DATE TIME' to datetime format, sort by 'DATE TIME', find the earlies date and only keep first 3 days of the data
    data['DATE TIME'] = pd.to_datetime(data['DATE TIME'])
    data = data.sort_values(by='DATE TIME')
    start_date = data['DATE TIME'].min()
    end_date = start_date + pd.DateOffset(days=1)
    data = data[(data['DATE TIME'] >= start_date) & (data['DATE TIME'] < end_date)]
    
    data.to_csv("data/case3_filtered.csv", index=False)
    
    data = pd.read_csv("data/case3_filtered.csv")

    # filter data to only keep column DIRECTION,TRIP ID,DATE TIME,STOP ID
    data = data[[time_column, stop_column, classification_column]]

    n_stops = data[stop_column].nunique()

    # make a new dataset, for each date, which stop id is included
    new_dataset = data.groupby(time_column)[stop_column].apply(list).reset_index()
    new_dataset.columns = ['time_value', 'stops']

    # add new column tp_value to the new_dataset, where tp_value = stops that is in the current timevalue / total num of stops
    new_dataset['tp_value'] = new_dataset['stops'].apply(lambda x: sum([x.count(stop_id) / n_stops for stop_id in x]))

    # only keep column event_date, tp_value  
    df = new_dataset[['time_value', 'tp_value']]

    # to csv, save in path 'data/transformed_symbseq.csv'
    df.to_csv("data/case3data.csv", index=False)

    return new_dataset, n_stops
