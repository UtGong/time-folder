import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from pandas import to_datetime

def generic_plot_setup(title, xlabel, ylabel, y_start, y_end, gap, dates=None, values=None, rotation=None, x_ticks=None, series_labels=None):
    plt.figure(figsize=(12, 6))

    if dates is not None and values is not None:
        # Convert dates to matplotlib date format if they're not already
        if not isinstance(dates[0], (float, int)):
            dates = mdates.date2num(dates)

        # Assuming values is a list of lists where each inner list is a series
        for index, series in enumerate(values):
            label = series_labels[index] if series_labels is not None and len(series_labels) > index else f"Series {index+1}"
            # Plot each series with a line and label for legend
            plt.plot(dates, series, label=label)
            # Fill the area under each line
            plt.fill_between(dates, series, alpha=0.3)  # Adjust alpha for fill transparency

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    plt.ylim(y_start, y_end)
    plt.yticks(np.arange(y_start, y_end + gap, step=gap))
    
    if rotation is not None and x_ticks is not None:
        # If x_ticks are provided and need rotation, adjust accordingly
        # This part might need refinement based on the specific datetime handling
        if not isinstance(x_ticks[0], (float, int)):
            x_ticks = mdates.date2num(x_ticks)
        plt.xticks(x_ticks, rotation=rotation)
    
    # Display the legend
    plt.legend()

    return plt


def draw_init_line_plot(data, date_column, value_column, y_start, y_end, gap):
    # Convert date column to datetime if not already done
    dates = to_datetime(data[date_column])
    # Ensure values is a list of numeric arrays
    values = [data[column].astype(float).values for column in value_column]
    return generic_plot_setup('Value Over Time', 'Date', 'Value', y_start, y_end, gap, dates, values, series_labels=value_column)

def nonlinear_plot_setup(title, xlabel, ylabel, y_start, y_end, gap, dates=None, values=None, rotation=None, x_ticks=None):
    plt.figure(figsize=(12, 6))

    if dates is not None and values is not None:
        # Use a range object as the x-axis values to distribute points evenly
        x_values = range(len(dates))

        # Plot each series with a line, using x_values for even distribution
        for series in values:
            plt.plot(x_values, series)
            # Fill the area under each line
            plt.fill_between(x_values, series, alpha=0.3)  # Adjust alpha for fill transparency

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    plt.ylim(y_start, y_end)
    plt.yticks(np.arange(y_start, y_end + gap, step=gap))
    
    # Set x-axis tick labels to the corresponding dates, rotated if specified
    if rotation is not None:
        plt.xticks(x_values, dates, rotation=rotation)
    else:
        plt.xticks(x_values, dates)  # Apply dates as labels without rotation

    return plt

def draw_merged_line_plot_non_linear(data, date_column, value_column, y_start, y_end, gap):
    # Convert each column into a list of floats for plotting
    values = [data[column].astype(float).values for column in value_column]
    
    # Get dates for tick labels
    dates = data[date_column].dt.strftime('%Y-%m-%d').tolist()  # Or any other string format you prefer
    
    # Call the generic plot setup with evenly distributed x-values and provided values
    plt_obj = nonlinear_plot_setup('Value Over Time', 'Date', 'Value', y_start, y_end, gap, dates, values, rotation=45)
    
    print("Merged data length: ", len(values))
    return plt_obj

def draw_merged_line_plot(data, y_start, y_end, gap):
    values = [point.start_point.data for point in data]
    values += data[-1].end_point.data
    dates = [to_datetime(point.start_point.time_value) for point in data]  
    print(to_datetime(data[-1].end_point.time_value))
    dates.append(to_datetime(data[-1].end_point.time_value))

    # Now call the generic plot setup with these datetime objects
    plt_obj = generic_plot_setup(
        'Value Over Time', 'Date', 'Value', y_start, y_end, gap, dates, values
    )
    
    # Adjust the x-axis to handle the datetime objects
    plt_obj.gca().xaxis_date()  # Interpret the x-axis values as dates
    plt_obj.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt_obj.gca().xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    plt_obj.gcf().autofmt_xdate()  # Rotate date labels automatically
    
    return plt_obj

def compute_y_axis_parameters(y_min, y_max):
    # Expand the range by 10% for aesthetic reasons
    range_expansion = 0.1 * (y_max - y_min)
    range_expansion = max(range_expansion, 0.1)  # Ensure at least a minimal expansion

    adjusted_min = y_min - range_expansion
    adjusted_max = y_max + range_expansion

    # Determine a suitable gap for the y-axis ticks
    ideal_range = adjusted_max - adjusted_min
    possible_gaps = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    gap = next((gap for gap in possible_gaps if ideal_range / gap <= 10), possible_gaps[-1])

    # Adjust start and end to align with the gap
    start = gap * round(adjusted_min / gap)
    end = gap * round(adjusted_max / gap + 0.5)  # Ensure rounding up for the end

    return start, end, gap
