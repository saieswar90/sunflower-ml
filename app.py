from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import os

app = Flask(__name__)

# Model configuration (relative path)
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "sunflower_leaf_classifier.tflite")

# Verify model exists
if not os.path.exists(TFLITE_MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {TFLITE_MODEL_PATH}")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class mapping based on training data
CLASS_LABELS = {
    0: "Leaf Scars",
    1: "Gray Mold",
    2: "Fresh Leaf",
    3: "Downy Mildew"
}

# Treatment suggestions
TREATMENT_SUGGESTIONS = {
    0: "Remove affected leaves and apply neem oil",
    1: "Use organic fungicide and improve airflow",
    2: "Plant is healthy - maintain regular care",
    3: "Apply copper-based fungicide and avoid wet foliage"
}

@app.route('/predict_sunflower', methods=['POST'])
def predict_sunflower():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Invalid file format'}), 400

    try:
        # Preprocess image
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((input_details[0]['shape'][2], input_details[0]['shape'][1]))
        img_array = np.array(img) / 255.0  # Adjust based on model's normalization
        img_array = np.expand_dims(img_array, axis=0).astype(input_details[0]['dtype'])

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        # Process results
        predicted_class = np.argmax(output)
        confidence = round(output[predicted_class] * 100, 2)
        
        return jsonify({
            'class_id': int(predicted_class),
            'class_name': CLASS_LABELS[predicted_class],
            'confidence': f"{confidence}%",
            'suggestion': TREATMENT_SUGGESTIONS[predicted_class]
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)