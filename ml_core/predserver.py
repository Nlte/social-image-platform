from flask import Flask, json, request, jsonify, render_template

import pdb
from server_model import ServerModel

app = Flask(__name__)

model = ServerModel()

@app.route('/')
def ping():
    return 'prediction server running\n'

@app.route('/predict', methods = ['POST'])
def predictions():
    if 'files' not in request.files:
        print('No file part')
        return ''
    fs = request.files['files'] # Flask FileStorage (.read() to get the byte content)
    encoded_image = fs.read()
    labels = model.inference(encoded_image)
    resp = {'labels':labels}
    print(resp)
    return jsonify(resp)


if __name__ == '__main__':
    # disable reloader to prevent loading the model each time
    app.run(host='127.0.0.1', port=8080, debug=True, use_reloader=False)
