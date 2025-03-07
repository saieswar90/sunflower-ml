import tensorflow as tf
import numpy as np
import cv2
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# Define the model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'sunflower_leaf_classifier.tflite')

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Labels and suggestions
labels = ["Leaf Scars", "Gray Mold", "Fresh Leaf", "Downy Mildew"]
suggestions = {
    "Leaf Scars": "Remove affected leaves and apply neem oil.",
    "Gray Mold": "Use organic fungicide and improve airflow.",
    "Fresh Leaf": "Plant is healthy - maintain regular care.",
    "Downy Mildew": "Apply copper-based fungicide and avoid wet foliage."
}

@app.route('/')
def home():
    return jsonify({"message": "Sunflower Disease Detection API is Running!"})

@app.route('/predict_sunflower', methods=['POST'])
def predict_disease():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    
    try:
        # Read and preprocess the image
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224)) / 255.0  # Resize and normalize
        image = np.expand_dims(image, axis=0).astype(np.float32)  # Add batch dimension
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Get prediction
        predicted_class = np.argmax(output_data)
        confidence = float(np.max(output_data))
        prediction = labels[predicted_class]
        suggestion = suggestions[prediction]

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "suggestion": suggestion
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
