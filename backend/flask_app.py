import io
import base64
from PIL import Image
import pandas as pd
import importlib
from flask_cors import CORS
from flask import Flask, request, jsonify

app = Flask(__name__)
CORS(app)

@app.route('/visualize', methods=['POST'])
def visualize():
    script_name = request.json['script_name']
    data_name = request.json['data_name']
    weight = request.json['weight']
    additional_params = request.json.get('additional_params', [])
    print("Additional params: ", additional_params)

    script_mappings = {
        'network_graph_main': ('network_graph_main', 'main'),
        'ss_main': ('ss_main', 'main'),
        'ts_main': ('ts_main', 'main')  # Handles both time series and stack graph
    }

    if script_name not in script_mappings:
        return jsonify({'error': 'Invalid script name provided'}), 400

    module_name, function_name = script_mappings[script_name]
    try:
        module = importlib.import_module(module_name)
        main_function = getattr(module, function_name)
    except (ImportError, AttributeError) as e:
        return jsonify({'error': 'Failed to load module or function', 'details': str(e)}), 500

    try:
        if additional_params:
            csv_file_path = data_name
            date_range = additional_params[0:2]  # This should be a list of two dates
            date_column = additional_params[2]
            value_column = additional_params[3]
            init_image_path, folded_image_path = main_function(csv_file_path, date_range, date_column, value_column, weight)
        else:
            init_image_path, folded_image_path = main_function(data_name, weight)

        init_image = Image.open(init_image_path)
        folded_image = Image.open(folded_image_path)

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
    app.run()
