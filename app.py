from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Class label mapping (update these if you know the actual class names)
CLASS_LABELS = {
    0: "Class 0",
    1: "Class 1",
    2: "Class 2"
}

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data.get("features")

        if not features:
            return jsonify({"error": "No features provided"}), 400

        if len(features) != 26:
            return jsonify({"error": f"Expected 26 features, got {len(features)}"}), 400

        input_array = np.array(features).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        probabilities = model.predict_proba(input_array)[0]

        return jsonify({
            "predicted_class": CLASS_LABELS.get(int(prediction), str(prediction)),
            "predicted_class_id": int(prediction),
            "probabilities": {
                CLASS_LABELS[i]: round(float(p), 4)
                for i, p in enumerate(probabilities)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "RandomForestClassifier"})


if __name__ == "__main__":
    app.run(debug=True)
