# ==========================================================
# Handwritten Digit Recognition using CNN
# Flask + TensorFlow
# ==========================================================

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# ----------------------------------------------------------
# Initialize Flask App
# ----------------------------------------------------------

app = Flask(__name__)

# ----------------------------------------------------------
# Load CNN Model
# ----------------------------------------------------------

try:
    model = load_model("model.keras")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Unable to load model:", e)
    model = None


# ----------------------------------------------------------
# Home Route
# ----------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "Project": "Handwritten Digit Recognition",
        "Framework": "Flask",
        "Model": "TensorFlow CNN",
        "Status": "Running Successfully"
    })


# ----------------------------------------------------------
# Prediction Route
# ----------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({
            "success": False,
            "message": "Model not loaded."
        }), 500

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "message": "No image uploaded."
        }), 400

    try:

        file = request.files["image"]

        image = Image.open(file).convert("L")

        image = image.resize((28, 28))

        image = np.array(image)

        image = image.astype("float32") / 255.0

        image = image.reshape(1, 28, 28, 1)

        prediction = model.predict(image, verbose=0)

        predicted_digit = int(np.argmax(prediction))

        confidence = round(float(np.max(prediction) * 100), 2)

        return jsonify({

            "success": True,
            "prediction": predicted_digit,
            "confidence": f"{confidence}%"

        })

    except Exception as e:

        return jsonify({

            "success": False,
            "message": str(e)

        }), 500


# ----------------------------------------------------------
# Health Check
# ----------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():

    return jsonify({

        "status": "healthy"

    })


# ----------------------------------------------------------
# Run Flask App
# ----------------------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
