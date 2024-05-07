import matplotlib.pyplot as plt

def plotting(df, unique_id, time_column, task_column):
    # Create a dictionary where keys are unique dates and values are lists of sub_IDs for that date
    date_to_subs = {row[time_column]: row[task_column] for index, row in df.iterrows()}

    # Plotting
    fig, ax = plt.subplots()
    for date, subs in date_to_subs.items():
        for sub in subs:
            # Find the index of the sub in unique_id to use as y-coordinate
            y = list(unique_id).index(sub)
            ax.plot(date, y, 'bo')  # 'bo' -> blue circle marker

    ax.set_yticks(range(len(unique_id)))
    ax.set_yticklabels(unique_id)
    ax.set_xlabel("Time Range")
    ax.set_ylabel("Unique ID")
    plt.xticks(rotation=45)
    plt.tight_layout()

    return plt

def plotting_nonlinear(df, unique_id, time_column, task_column):
    # Sort the DataFrame by the time column to ensure the dates are in order
    df = df.sort_values(by=time_column)
    
    # Create a list of unique, sorted dates
    dates = sorted(set(df[time_column]))
    
    # Map these dates to a range of integers (x_values)
    date_to_index = {date: i for i, date in enumerate(dates)}
    
    # Adjusted plotting to use uniform x-axis values
    fig, ax = plt.subplots()
    for index, row in df.iterrows():
        date, subs = row[time_column], row[task_column]
        x = date_to_index[date]
        for sub in subs:
            y = list(unique_id).index(sub)
            ax.plot(x, y, 'bo')  # 'bo' -> blue circle marker
    
    # Set the y-axis with the unique IDs
    ax.set_yticks(range(len(unique_id)))
    ax.set_yticklabels(unique_id)
    
    # Adjusting x-axis to use uniform integer values and setting custom tick labels
    ax.set_xticks(range(len(dates)))  # Set x-ticks positions
    
    # Labeling the axes
    ax.set_xlabel("Time Range")
    ax.set_ylabel("Unique ID")
    
    plt.tight_layout()
    return plt