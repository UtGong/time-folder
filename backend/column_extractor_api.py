from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This line enables CORS for all domains on all routes

@app.route('/get-columns', methods=['POST'])
def get_columns():
    # Extract the file path from the request JSON
    data_path = request.json.get('dataPath')
    if not data_path:
        return jsonify({"error": "Missing data path"}), 400
    
    try:
        # Load the CSV file using Pandas
        df = pd.read_csv(data_path)
        # Retrieve all column names except the first one
        columns = df.columns.tolist()[1:]  # Skip the first column
        return jsonify({"columns": columns}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
