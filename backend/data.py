class TimePoint:
    def __init__(self, idx, time_value, data):
        self.idx = idx
        self.time_value = time_value
        self.data = data


class TimeFrame:
    def __init__(self, start_point, end_point):
        self.start_point = start_point
        self.end_point = end_point


def build_time_frames(dataset, date_column, value_column):
    """
    Builds a list of time frames from given data.

    Parameters:
    - data: The dataset from which to build the time frames.
    - date_column: The name of the column containing date information.
    - value_column: The name of the column containing value information.

    Returns:
    - A list of Node objects each representing a time frame.
    """
    time_points = []
    time_frames = []
    for i in range(len(dataset)):
        time_value = dataset.iloc[i][date_column]
        data = []
        for j in range(len(value_column)):
            data.append(dataset.iloc[i][value_column[j]])
        time_point = TimePoint(idx=i, data=data, time_value=time_value)
        time_points.append(time_point)
    for i in range(len(dataset) - 1):
        time_frame = TimeFrame(time_points[i], time_points[i+1])
        time_frames.append(time_frame)
    return time_frames


def combine_time_frames(tfs):
    start_point = tfs[0].start_point
    end_point = tfs[-1].end_point
    return [TimeFrame(start_point, end_point)]
