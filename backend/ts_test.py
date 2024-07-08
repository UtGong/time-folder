import subprocess

def run_test():
    parameters = {
        "csv_file_path": "data/StnData_2020-2023_dailytemp.csv",
        "date_range": ["2020-01-01", "2023-12-31"],
        "date_column": "Date",
        "value_column": ["Max"],
        "weight": 3.0
    }

    csv_file_path = parameters["csv_file_path"]
    date_range = parameters["date_range"]
    date_column = parameters["date_column"]
    value_column = parameters["value_column"]
    weight = parameters["weight"]

    load_args = [
        "python", "ts_main.py",
        csv_file_path,
        date_range[0],
        date_range[1],
        date_column,
        value_column[0],
        str(weight)
    ]

    load_process = subprocess.run(load_args, capture_output=True, text=True)
    if load_process.returncode != 0:
        print("Error in ts_main.py:")
        print(load_process.stderr)
        return
    else:
        print("ts_main.py output:")
        print(load_process.stdout)

    visualize_args = [
        "python", "ts_visualize.py"
    ]

    visualize_process = subprocess.run(visualize_args, capture_output=True, text=True)
    if visualize_process.returncode != 0:
        print("Error in ts_visualize.py:")
        print(visualize_process.stderr)
    else:
        print("ts_visualize.py output:")
        print(visualize_process.stdout)

if __name__ == "__main__":
    run_test()
