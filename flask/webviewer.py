import sys
import os
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flask import Flask, jsonify, render_template, request, redirect, url_for
from flask_cors import CORS
from helpers.helpers import hash_params


app = Flask(__name__)

CORS(app)  # Enable CORS for all routes



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data as dictionary, flat=False preserves multiple values per key
        form_data = request.form.to_dict(flat=True)
        items = list(form_data.items())
        items.insert(3, ("smids_input_path", "./data/final_input.csv"))
        items.insert(4, ("ground_truth_path", "./data/data_raw.csv"))
        form_data = dict(items)


        experiment_hash = str(hash_params(form_data))

        print(experiment_hash)
        print(form_data)

        
        return render_template('form.html', submitted=True, data=form_data)
    return render_template('form.html', submitted=False)

@app.route('/get_hash', methods=['POST'])
def get_hash():
    print("got post")
    json = request.get_json()
    experiment_hash = str(hash_params(json))
    return jsonify({
        'redirect_url': url_for('get_results', experiment_hash=experiment_hash)
    })

@app.route('/get_results/<experiment_hash>')
def get_results(experiment_hash):
    print("got get results")
    return render_template('results.html', experiment_hash=experiment_hash) 

if __name__ == '__main__':
    app.run(debug=True)