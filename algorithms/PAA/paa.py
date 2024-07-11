import pandas as pd
import matplotlib.pyplot as plt
import math

def paa(ts, paa_rate, alphabet_size=4):
    length = len(ts)
    paa_size = math.floor(length * paa_rate)

    if length == paa_size:
        return ts
    else:
        ret = [0 for x in range(paa_size)]
        pos = 0
        for i in range(0, length * paa_size, paa_size):
            if i >= pos and i + paa_size <= pos + length:
                ret[pos // length] += ts[i // paa_size] * paa_size
            elif i + paa_size > pos + length:
                ret[pos // length] += ts[i // paa_size] * (pos + length - i)
                ret[pos // length + 1] += ts[i // paa_size] * (i + paa_size - pos - length)
                pos += length

        for i in range(0, paa_size):
            ret[i] /= length
        return ret

def save_paa_ts_plot():

    # Load the CSV file
    file_path = '../../backend\data\StnData_2020-2023_dailytemp.csv'  # Update this to your file path
    data = pd.read_csv(file_path)

    # Define the PAA rate and output filename
    paa_rate = 0.016  # Example rate
    filename = 'paa_ts.png'
    # Extract the 'Max' column
    max_temps = data['Max'].tolist()

    # Apply the PAA function
    paa_result = paa(max_temps, paa_rate)

    # print("paa_result", paa_result)
    # print("Original data", max_temps)

    # Generate x-values for the PAA result to match the reduced segments
    x_values = [i * (len(max_temps) / len(paa_result)) for i in range(len(paa_result))]

    # Plot the original Max temperatures
    plt.figure(figsize=(14, 7))

    # Plot the PAA reduced Max temperatures directly
    plt.plot(x_values, paa_result, label='PAA Reduced Max Temperatures', linewidth=2, color='red')
    plt.xlabel('Day')
    plt.ylabel('Max Temperature')
    plt.title('Original vs PAA Reduced Max Temperatures')
    plt.legend()

    # Save the plot to a file
    plt.savefig(filename)
    plt.close()

# ts_plot 
save_paa_ts_plot()
print(f'TS plot saved as paa_ts.png')



