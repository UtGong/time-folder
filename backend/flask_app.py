import io
import base64
from PIL import Image
import pandas as pd
import importlib
import sys
from flask_cors import CORS
from flask import Flask, request, jsonify

sys.path.append('/home/jojogong3736/.virtualenvs/my-virtualenv/lib/python3.9/site-packages')

app = Flask(__name__)
CORS(app)

script_mappings = {
    'network_graph_main': ('network_graph_main', 'run_algorithm', 'network_graph_visualize', 'visualize_results'),
    'ss_main': ('ss_main', 'run_algorithm', 'ss_visualize', 'visualize_results'),
    'ts_main': ('ts_main', 'run_algorithm', 'ts_visualize', 'visualize_results')  # Handles both time series and stack graph
}
@app.route('/run-algorithm', methods=['POST'])
def run_algorithm():
    script_name = request.json['script_name']
    data_name = request.json['data_name']
    weight = request.json['weight']
    additional_params = request.json.get('additional_params', [])
    print("Additional params: ", additional_params)

    if script_name not in script_mappings:
        return jsonify({'error': 'Invalid script name provided'}), 400

    module_name, function_name, _, _ = script_mappings[script_name]
    try:
        module = importlib.import_module(module_name)
        main_function = getattr(module, function_name)
        print("------ main function got")
    except (ImportError, AttributeError) as e:
        return jsonify({'error': 'Failed to load module or function', 'details': str(e)}), 500

    try:
        if additional_params:
            csv_file_path = data_name
            date_range = additional_params[0:2]
            date_column = additional_params[2]
            value_column = additional_params[3]
            print(f"Running with value_column: {value_column}")
            dataset, data, date_column, value_column, runtime, original_length, folded_length = main_function(csv_file_path, date_range, date_column, value_column, weight)
            dataset.to_pickle('dataset.pkl')
            data.to_pickle('data.pkl')
            with open('meta.txt', 'w') as f:
                f.write(f"{date_column}," + ",".join(value_column))
        else:
            print(f"Running {script_name} without additional params")
            dataset, data, output_path, runtime, original_length, folded_length = main_function(data_name, weight)
            dataset.to_pickle('dataset.pkl')
            data.to_pickle('data.pkl')
            with open('meta.txt', 'w') as f:
                f.write(f"{script_name},{data_name},{weight}")

        return jsonify({
            'runtime': runtime,
            'original_length': original_length,
            'folded_length': folded_length
        })

    except Exception as e:
        return jsonify({'error': 'An error occurred while running the algorithm', 'details': str(e)}), 500

@app.route('/visualize', methods=['POST'])
def visualize():
    script_name = request.json['script_name']
    if script_name not in script_mappings:
        return jsonify({'error': 'Invalid script name provided'}), 400

    _, _, module_name, function_name = script_mappings[script_name]
    try:
        module = importlib.import_module(module_name)
        visualize_function = getattr(module, function_name)
        print("------ visualize function got")
    except (ImportError, AttributeError) as e:
        return jsonify({'error': 'Failed to load module or function', 'details': str(e)}), 500

    try:
        init_graph_path, folded_graph_path = visualize_function()

        init_image = Image.open(init_graph_path)
        folded_image = Image.open(folded_graph_path)

        init_image_bytes = io.BytesIO()
        folded_image_bytes = io.BytesIO()

        init_image.save(init_image_bytes, format='PNG')
        folded_image.save(folded_image_bytes, format='PNG')

        init_image_b64 = base64.b64encode(init_image_bytes.getvalue()).decode('utf-8')
        folded_image_b64 = base64.b64encode(folded_image_bytes.getvalue()).decode('utf-8')

        return jsonify({
            'init_image': f'data:image/png;base64,{init_image_b64}',
            'folded_image': f'data:image/png;base64,{folded_image_b64}'
        })

    except Exception as e:
        return jsonify({'error': 'An error occurred while generating the visualization', 'details': str(e)}), 500


@app.route('/get-columns', methods=['POST'])
def get_columns():
    data_path = request.json.get('dataPath')
    if not data_path:
        return jsonify({"error": "Missing data path"}), 400

    try:
        df = pd.read_csv(data_path)
        columns = df.columns.tolist()[1:]  # Skip the first column
        return jsonify({"columns": columns}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
