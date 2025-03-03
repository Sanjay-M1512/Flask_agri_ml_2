import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

# Load the trained model
MODEL_PATH = "pesticide_recommendation_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels and pesticide mapping
class_labels = [
    "Pepper__bell___Bacterial_spot",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Tomato_Target_Spot",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_Tomato_YellowLeaf_Curl",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted",
]

pesticide_mapping = {
    "Pepper__bell___Bacterial_spot": "Copper-based fungicides, Streptomycin",
    "Potato___Early_blight": "Chlorothalonil, Mancozeb",
    "Potato___Late_blight": "Metalaxyl, Mancozeb",
    "Tomato_Target_Spot": "Chlorothalonil, Azoxystrobin",
    "Tomato_Tomato_mosaic_virus": "No chemical treatment, use resistant varieties",
    "Tomato_Tomato_YellowLeaf_Curl": "Insecticides for whitefly control (Imidacloprid, Spinosad)",
    "Tomato_Bacterial_spot": "Copper-based fungicides, Streptomycin",
    "Tomato_Early_blight": "Mancozeb, Chlorothalonil",
    "Tomato_Leaf_Mold": "Sulfur-based fungicides, Chlorothalonil",
    "Tomato_Septoria_leaf_spot": "Copper-based fungicides, Chlorothalonil",
    "Tomato_Spider_mites_Two_spotted": "Abamectin, Spinosad",
}

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/pest', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Preprocess image and make prediction
    img_array = preprocess_image(file_path)
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    
    # Get pesticide recommendation
    pesticide = pesticide_mapping.get(predicted_class, "No recommendation available")
    
    return jsonify({
        "disease": predicted_class,
        "pesticide_recommendation": pesticide
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
