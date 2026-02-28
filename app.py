from flask import Flask, request, jsonify, render_template_string
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

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>ML Model Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 700px; margin: 50px auto; padding: 20px; }
        h2 { color: #333; }
        textarea { width: 100%; height: 80px; padding: 10px; font-size: 14px; }
        button { background: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; font-size: 16px; border-radius: 5px; margin-top: 10px; }
        button:hover { background: #45a049; }
        #result { margin-top: 20px; padding: 15px; background: #f0f0f0; border-radius: 5px; display: none; }
        .label { font-weight: bold; color: #333; }
        .hint { font-size: 12px; color: #888; }
    </style>
</head>
<body>
    <h2> ML Model Predictor</h2>
    <p>Model: <b>Random Forest Classifier</b> | Accuracy: <b>90.7%</b> | Classes: <b>0, 1, 2</b></p>

    <label class="label">Enter 26 feature values (comma-separated):</label><br>
    <p class="hint">Example: 1.2, 3.4, 0.5, 2.1, ...</p>
    <textarea id="features" placeholder="e.g. 1.2, 3.4, 0.5, 2.1, 1.0, 0.8, ..."></textarea><br>
    <button onclick="predict()">Predict</button>

    <div id="result">
        <p><span class="label">Predicted Class:</span> <span id="pred_class"></span></p>
        <p><span class="label">Probabilities:</span> <span id="pred_proba"></span></p>
    </div>

    <script>
        async function predict() {
            const raw = document.getElementById("features").value;
            const features = raw.split(",").map(Number);

            if (features.length !== 26) {
                alert("Please enter exactly 26 feature values.");
                return;
            }

            const res = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ features: features })
            });

            const data = await res.json();
            if (data.error) {
                alert("Error: " + data.error);
                return;
            }

            document.getElementById("pred_class").innerText = data.predicted_class;
            document.getElementById("pred_proba").innerText = JSON.stringify(data.probabilities);
            document.getElementById("result").style.display = "block";
        }
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)


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