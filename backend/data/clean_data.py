import pandas as pd
import numpy as np

data = pd.read_csv('data/wfs_behaviors_and_records_508p-546d-98r_20220722173739.csv', encoding='latin1', low_memory=False)

data['event_date'] = pd.to_datetime(data['event_date'])

# only keep column sub_ID and event date
data = data[['sub_ID', 'event_date', 'behav_comptype_h']]

# filter data by 2021.01.01 < event date < 2021.01.31 and sub_ID < 98000011
data = data[(data['event_date'] > '2021-01-01') & (data['event_date'] < '2022-01-01') & (data['sub_ID'] < 98000015)]

# filter data, for each duplicate unique id + event, randomly choose a single row to keep
data = data.groupby(['sub_ID', 'event_date'], group_keys=False).apply(lambda x: x.sample(1))

# random sample len(data)*0.8 rows
data = data.sample(frac=0.7)

# reset the index to avoid the mentioned ambiguity
data.reset_index(drop=True, inplace=True)

# sort data by date
data = data.sort_values(by='event_date')

# save to csv file
data.to_csv('data/symbolic_seq.csv', index=False)
