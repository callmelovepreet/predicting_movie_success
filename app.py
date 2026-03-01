from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ── Correct class label mapping ──
# LabelEncoder encodes alphabetically: Average=0, Flop=1, Hit=2
CLASS_LABELS = {
    0: "Average",
    1: "Flop",
    2: "Hit"
}

# ── Approximate StandardScaler statistics from the training dataset ──
# Feature order (26 features, same as notebook after preprocessing):
# color, num_critic_for_reviews, duration, director_facebook_likes,
# actor_3_facebook_likes, gross, genre_type1..8,
# num_voted_users, cast_total_facebook_likes, facenumber_in_poster,
# num_user_for_reviews, language, country, content_rating,
# budget, title_year, actor_2_facebook_likes, aspect_ratio, movie_facebook_likes

FEATURE_MEANS = np.array([
    0.91,        # color
    140.0,       # num_critic_for_reviews
    107.0,       # duration
    686.0,       # director_facebook_likes
    645.0,       # actor_3_facebook_likes
    48468412.0,  # gross
    6.0,         # genre_type1
    2.0,         # genre_type2
    1.5,         # genre_type3
    0.5,         # genre_type4
    0.2,         # genre_type5
    0.1,         # genre_type6
    0.05,        # genre_type7
    0.02,        # genre_type8
    83668.0,     # num_voted_users
    9699.0,      # cast_total_facebook_likes
    1.37,        # facenumber_in_poster
    272.0,       # num_user_for_reviews
    10.5,        # language
    58.0,        # country
    8.5,         # content_rating
    39752624.0,  # budget
    2002.0,      # title_year
    1652.0,      # actor_2_facebook_likes
    2.05,        # aspect_ratio
    7525.0       # movie_facebook_likes
])

FEATURE_STDS = np.array([
    0.28,        # color
    121.0,       # num_critic_for_reviews
    26.0,        # duration
    2813.0,      # director_facebook_likes
    1665.0,      # actor_3_facebook_likes
    68237508.0,  # gross
    4.5,         # genre_type1
    3.5,         # genre_type2
    2.8,         # genre_type3
    1.5,         # genre_type4
    0.8,         # genre_type5
    0.5,         # genre_type6
    0.3,         # genre_type7
    0.15,        # genre_type8
    138490.0,    # num_voted_users
    18163.0,     # cast_total_facebook_likes
    1.95,        # facenumber_in_poster
    377.0,       # num_user_for_reviews
    5.5,         # language
    10.0,        # country
    4.5,         # content_rating
    60000000.0,  # budget
    12.5,        # title_year
    6183.0,      # actor_2_facebook_likes
    0.4,         # aspect_ratio
    19320.0      # movie_facebook_likes
])


def scale_features(features):
    """Apply StandardScaler normalization (same as training pipeline)."""
    arr = np.array(features, dtype=float)
    return (arr - FEATURE_MEANS) / FEATURE_STDS


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

        # Scale features before passing to model (model was trained on scaled data)
        scaled = scale_features(features).reshape(1, -1)

        prediction = model.predict(scaled)[0]
        probabilities = model.predict_proba(scaled)[0]

        pred_label = CLASS_LABELS.get(int(prediction), str(prediction))

        return jsonify({
            "predicted_class": pred_label,
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
    return jsonify({
        "status": "ok",
        "model": "RandomForestClassifier (GridSearchCV)",
        "classes": list(CLASS_LABELS.values())
    })


if __name__ == "__main__":
    app.run(debug=True)
