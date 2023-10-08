from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import matplotlib.pyplot as plt
from PIL import Image
from source import preprocess_input_image, batch_predict, conv_float_int, combine_image, load_trained_model, burn_area   
import numpy as np
from keras.models import load_model
import tempfile

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
model = load_trained_model("temp_model.h5")

@app.route('/process-image', methods=['POST'])
def process_image():
    uploaded_file = request.files['file']
    if uploaded_file:
        uploaded_image = Image.open(uploaded_file)
        uploaded_image = uploaded_image.convert("RGB")
        input_image_array = np.array(uploaded_image)
        original_width, original_height, pix_num = input_image_array.shape
        new_image_array, row_num, col_num = preprocess_input_image(input_image_array)
        preds = batch_predict(new_image_array, model)
        #preds_t = (preds > 0.25).astype(np.uint8)
        output_pred = conv_float_int(combine_image(preds, row_num, col_num, original_width, original_height, remove_ghost=True)[:,:,0])
        #output_mask = conv_float_int(combine_image(preds_t, row_num, col_num, original_width, original_height, remove_ghost=False)[:,:,0])
        output_pred = output_pred.astype(np.uint8)
        # Save the processed image to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        img = Image.fromarray(output_pred)
        img.save(temp_file.name)

        return send_file(temp_file.name, mimetype='image/png')

@app.route('/coordinates', methods=['GET'])
def coordinates():
    return jsonify({
        "north": 44.130045,
        "south": 44.0,
        "east": -65.9,
        "west": -66.05725,
    })

if __name__ == '__main__':
    app.run(port=5000)
