import pandas as pd
import matplotlib.pyplot as plt


def data_transformation(data_name, time_column, user_column, status_column):
    file_path = "data/" + data_name + ".csv"
    df = pd.read_csv(file_path)

    # sort by time_column and only keep the first 10000 rows
    df = df.sort_values(by=[time_column]).head(1000000)

    unique_id_num = df[user_column].nunique()
    unique_id = df[user_column].unique()

    # filter df to only keep 100 unique value of user_column
    df = df[df[user_column].isin(unique_id[:30])]

    # filter the rows that has the lowest 10 value in the column time_column
    df = df[df[time_column] > df[time_column].quantile(0.1)]

    # Assuming df is your DataFrame
    grouped = df.groupby(time_column)

    def transform_group(group):
        return [{'user_id': row[user_column], 'task_status': row[status_column]} for index, row in group.iterrows()]

    transformed_df = grouped.apply(transform_group).reset_index()
    transformed_df.columns = ['time_value', 'user_task_status']

    # add new column tp_value to the transformed_df, where tp_value = sum(for each type of task_status, count the number of users that have that task_status at that time/total number of users), do not use lambda
    transformed_df['tp_value'] = transformed_df['user_task_status'].apply(lambda x: sum([x.count(task_status) / unique_id_num for task_status in x]))
    # only keep column event_date, tp_value
    df = transformed_df[['time_value', 'tp_value']]

    # to csv, save in path 'data/transformed_symbseq.csv'
    df.to_csv("data/ss_to_ts.csv", index=False)

    return transformed_df, unique_id

def transform_HR_data():
    data = pd.read_csv("/home/jojogong3736/mysite/backend/data/HR.csv")

    # drop first column
    df = data.drop(data.columns[0], axis=1)
    unique_id = range(0, 15)
    unique_id_num = 15

    # for each cell in each columm each row, if the value is after 10:00:00, turn it into categories 'late', if before 10:00:00, 'on time', if missing value, 'Absance'
    # Function to categorize each cell
    def categorize_time(time):
        if pd.isnull(time):
            return 'Absence'
        time_part = pd.to_datetime(time).time()
        if time_part >= pd.to_datetime('10:00:00').time():
            return 'late'
        else:
            return 'on time'

    # Apply the function to each cell in the DataFrame
    for col in df.columns:
        df[col] = df[col].apply(categorize_time)

    # Initialize an empty list to hold the dictionaries
    data_list = []

    # Iterate over the columns in the DataFrame
    for column in df.columns:
        # Skip the 'Unnamed: 0' column if it exists
        if column == 'Unnamed: 0':
            continue

        # Create a dictionary for the current column and append it to the list
        status_list = df[column].tolist()
        data_list.append({'time_value': column, 'user_task_status': status_list})

    # Create a new DataFrame from the list of dictionaries
    new_dataset = pd.DataFrame(data_list)

    # add new column tp_value to the transformed_df, where tp_value = sum(for each type of task_status, count the number of users that have that task_status at that time/total number of users), do not use lambda
    new_dataset['tp_value'] = new_dataset['user_task_status'].apply(lambda x: sum([x.count(task_status) / unique_id_num for task_status in x]))
    # only keep column event_date, tp_value
    df = new_dataset[['time_value', 'tp_value']]

    # to csv, save in path 'data/transformed_symbseq.csv'
    df.to_csv("/home/jojogong3736/mysite/backend/data/ss_to_ts.csv", index=False)

    return new_dataset, unique_id