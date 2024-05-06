import math

WEIGHT = 200

def calculate_model_dl(tfs):
    T = tfs[-1].end_point.idx - tfs[0].start_point.idx + 1
    n = 2
    return (T - n) / 2 * math.log2(T - n + 1)

def calculate_data_dl(tfs, weight=WEIGHT):
    # line_num = len(tfs[-1].end_point.data)
    # total_value = 0
    # for i in range(line_num):
    #     slope = tfs[-1].end_point.data[i] - tfs[0].start_point.data[i]
    #     encoded_value = 1 / (1 + math.exp(-slope))
    #     total_value += encoded_value
    # value = total_value / line_num
    # value = (tfs[0].start_point.data[0] + tfs[-1].end_point.data[0])/2
    value = tfs[0].start_point.data[0]
    encoded_value = 1 / (1 + math.exp(-value))
    encoded_length = -weight * math.log2(encoded_value)
    return encoded_length

def calculate_total_dl(tfs, weight=WEIGHT):
    return calculate_model_dl(tfs) + calculate_data_dl(tfs, weight)