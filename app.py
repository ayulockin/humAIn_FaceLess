from flask import Flask, request, jsonify
import numpy as np
import base64
from io import BytesIO

from main import working

app = Flask(__name__)

@app.route('/')
def hello():
    return 'TCS humAIn'

@app.route('/predict', methods=['POST'])
def predict():
    input_file = request.files.get('file')
    if not input_file:
        return "File is not present in the request"
    if input_file.filename == '':
        return "Filename is not present in the request"
    if not input_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return "Invalid file type"
    else:
        return jsonify(working(input_file)) 

if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 8000
    app.run(port=port, debug=True)