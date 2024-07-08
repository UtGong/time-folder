import subprocess

def run_test():
    parameters = {
        "script_name": "ss_main",
        "data_name": "HR",
        "weight": 5.0
    }

    script_name = parameters["script_name"]
    data_name = parameters["data_name"]
    weight = parameters["weight"]

    load_args = [
        "python", "ss_main.py",
        data_name,
        str(weight)
    ]

    load_process = subprocess.run(load_args, capture_output=True, text=True)
    if load_process.returncode != 0:
        print("Error in ss_main.py:")
        print(load_process.stderr)
        return
    else:
        print("ss_main.py output:")
        print(load_process.stdout)

    visualize_args = [
        "python", "ss_visualize.py"
    ]

    visualize_process = subprocess.run(visualize_args, capture_output=True, text=True)
    if visualize_process.returncode != 0:
        print("Error in ss_visualize.py:")
        print(visualize_process.stderr)
    else:
        print("ss_visualize.py output:")
        print(visualize_process.stdout)

if __name__ == "__main__":
    run_test()
