from flask import Flask, request, jsonify
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the model
model = load_model('Image_classify_2.keras')  # Replace with the actual path to your model file

def preprocess_image(image):
    image = image.resize((224, 224))  # Example size; adjust as needed
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def home():
    return "Welcome to the Image Classification API!"

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    try:
        file = request.files['file']
        image = Image.open(file)
        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
