import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Upload folder setup
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Model path
MODEL_PATH = "pesticide_recommendation_model.h5"

# Allowed image formats
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Class labels and pesticide recommendations
class_labels = [
    "Pepper__bell___Bacterial_spot", "Potato___Early_blight", "Potato___Late_blight",
    "Tomato_Target_Spot", "Tomato_Tomato_mosaic_virus", "Tomato_Tomato_YellowLeaf_Curl",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted"
]

pesticide_mapping = {
    "Pepper__bell___Bacterial_spot": "Copper-based fungicides, Streptomycin",
    "Potato___Early_blight": "Chlorothalonil, Mancozeb",
    "Potato___Late_blight": "Metalaxyl, Mancozeb",
    "Tomato_Target_Spot": "Chlorothalonil, Azoxystrobin",
    "Tomato_Tomato_mosaic_virus": "No chemical treatment, use resistant varieties",
    "Tomato_Tomato_YellowLeaf_Curl": "Imidacloprid, Spinosad (for whitefly control)",
    "Tomato_Bacterial_spot": "Copper-based fungicides, Streptomycin",
    "Tomato_Early_blight": "Mancozeb, Chlorothalonil",
    "Tomato_Leaf_Mold": "Sulfur-based fungicides, Chlorothalonil",
    "Tomato_Septoria_leaf_spot": "Copper-based fungicides, Chlorothalonil",
    "Tomato_Spider_mites_Two_spotted": "Abamectin, Spinosad",
}

# Load model once at startup
try:
    logging.info("🔄 Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info("✅ Model loaded successfully!")
except Exception as e:
    logging.error(f"❌ Model loading failed: {e}")
    model = None

def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for model prediction."""
    try:
        img = load_img(image_path, target_size=(224, 224))  # Resize for MobileNetV2
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        return np.expand_dims(img_array.astype(np.float32), axis=0)  # Add batch dimension
    except Exception as e:
        logging.error(f"❌ Error processing image: {e}")
        return None

@app.route("/pest", methods=["POST"])
def predict():
    """Handle pesticide recommendation prediction."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file format. Only PNG, JPG, and JPEG allowed"}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        logging.info(f"📥 File received: {filename}")

        # Preprocess image
        img_array = preprocess_image(file_path)
        if img_array is None:
            return jsonify({"error": "Failed to process image"}), 500

        # Ensure model is loaded
        if model is None:
            return jsonify({"error": "Model not available"}), 500

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions)]

        # Get pesticide recommendation
        pesticide = pesticide_mapping.get(predicted_class, "No recommendation available")

        logging.info(f"🌿 Disease detected: {predicted_class}")
        logging.info(f"🧴 Pesticide recommended: {pesticide}")

        # Remove uploaded file after processing
        os.remove(file_path)

        return jsonify({
            "disease": predicted_class,
            "pesticide_recommendation": pesticide
        })

    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=True)
