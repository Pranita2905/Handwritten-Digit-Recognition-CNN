# ==========================================================
# Handwritten Digit Recognition using CNN (Flask + TensorFlow)
# Model : TensorFlow / Keras Sequential CNN
# ==========================================================

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# ==========================================================
# Flask Configuration
# ==========================================================

app = Flask(__name__)

# ==========================================================
# Load Trained Model
# ==========================================================

MODEL_PATH = "model.keras"

try:
    model = load_model(MODEL_PATH)
    print("✅ CNN Model Loaded Successfully")
except Exception as e:
    print(f"❌ Failed to Load Model: {e}")
    model = None

# ==========================================================
# Home Route
# ==========================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "Project": "Handwritten Digit Recognition",
        "Framework": "Flask",
        "Deep Learning": "TensorFlow / Keras CNN",
        "Status": "Running Successfully 🚀"
    })

# ==========================================================
# Prediction Route
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({
            "error": "Model not loaded."
        }), 500

    if "image" not in request.files:
        return jsonify({
            "error": "Please upload an image."
        }), 400

    try:

        image = Image.open(request.files["image"]).convert("L")

        image = image.resize((28, 28))

        image = np.array(image)

        image = image.astype("float32") / 255.0

        image = image.reshape(1, 28, 28, 1)

        prediction = model.predict(image)

        predicted_digit = int(np.argmax(prediction))

        confidence = float(np.max(prediction) * 100)

        return jsonify({

            "Predicted Digit": predicted_digit,
            "Confidence": f"{confidence:.2f}%"

        })

    except Exception as e:

        return jsonify({

            "error": str(e)

        }), 500

# ==========================================================
# Health Check
# ==========================================================

@app.route("/health")
def health():

    return jsonify({

        "status": "Healthy ✅"

    })

# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(

        host="0.0.0.0",
        port=port,
        debug=False

    )
