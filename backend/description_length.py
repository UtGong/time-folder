import math

WEIGHT = 3

def calculate_model_dl(tfs):
    T = tfs[-1].end_point.idx - tfs[0].start_point.idx + 1
    n = 2
    return (T - n) / 2 * math.log2(T - n + 1)

def calculate_data_dl(tfs, weight):
    line_num = len(tfs[-1].end_point.data)
    total_value = 0
    for i in range(line_num):
        slope = (tfs[-1].end_point.data[i] - tfs[0].start_point.data[i]) / 100
        encoded_value = 1 / (1 + math.exp(-slope))
        total_value += encoded_value
    value = total_value / line_num
    encoded_length = -weight * math.log2(value)
    return encoded_length

def calculate_total_dl(tfs, weight):
    return calculate_model_dl(tfs) + calculate_data_dl(tfs, weight)