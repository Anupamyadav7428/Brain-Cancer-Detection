from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load your trained model
model_path = "model/model.h5"  # Adjust the path according to your directory structure
model = tf.keras.models.load_model(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Preprocess the image
    resized_img = cv2.resize(img, (256, 256))
    scaled_img = resized_img / 255.0
    input_img = np.expand_dims(scaled_img, axis=0)
    
    # Make prediction
    prediction = model.predict(input_img)
    
    # Return the result
    if prediction > 0.5:
        result = 'Brain with Tumor'
    else:
        result = 'Brain without Tumor'
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)