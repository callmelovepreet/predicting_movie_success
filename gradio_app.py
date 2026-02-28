# Movie Success Prediction - Gradio Interface
import gradio as gr
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')

# Global variables to store model and preprocessors
model = None
scaler = None
feature_columns = None
label_encoders = {}
target_encoder = None
categorical_cols = []
numerical_cols = []

def load_and_explore_data(filepath):
    db = pd.read_csv(filepath)
    return db

def create_target_variable(db):
    def classify_movie(score):
        if pd.isna(score):
            return np.nan
        elif 1 <= score < 3:
            return 'Flop'
        elif 3 <= score < 6:
            return 'Average'
        elif 6 <= score <= 10:
            return 'Hit'
        else:
            return np.nan
    
    db['Classify'] = db['imdb_score'].apply(classify_movie)
    return db

def preprocess_data(db):
    global categorical_cols, numerical_cols, label_encoders, target_encoder
    
    db_processed = db.copy()
    
    columns_to_drop = [
        'movie_title', 'director_name', 'actor_1_name', 'actor_2_name',
        'actor_3_name', 'plot_keywords', 'movie_imdb_link','actor_1_facebook_likes','imdb_score','color', 
        'genre_type2', 'genre_type3', 'genre_type4', 'genre_type5', 'genre_type6', 'genre_type7', 'genre_type8'
    ]
    db_processed = db_processed.drop(columns=columns_to_drop, errors='ignore')
    
    X = db_processed.drop('Classify', axis=1)
    y = db_processed['Classify']
    
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    num_imputer = SimpleImputer(strategy='median')
    X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
    
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    return X, y_encoded, target_encoder

def train_model_pipeline(filepath):
    """Train the model and return all necessary components"""
    global model, scaler, feature_columns
    
    # Load and prepare data
    db = load_and_explore_data(filepath)
    db = create_target_variable(db)
    X, y_encoded, target_enc = preprocess_data(db)
    
    feature_columns = X.columns.tolist()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train optimized model
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    print("Model training complete!")
    return model, scaler, feature_columns, target_enc

def predict_movie_success(
    num_critic_for_reviews,
    duration,
    director_facebook_likes,
    actor_3_facebook_likes,
    actor_2_facebook_likes,
    gross,
    genre_type1,
    num_voted_users,
    cast_total_facebook_likes,
    facenumber_in_poster,
    num_user_for_reviews,
    language,
    country,
    content_rating,
    budget,
    title_year,
    aspect_ratio,
    movie_facebook_likes
):
    """Make prediction based on user inputs"""
    
    if model is None:
        return "❌ Model not trained yet. Please train the model first."
    
    try:
        # Create input dataframe
        input_data = {
            'num_critic_for_reviews': [num_critic_for_reviews],
            'duration': [duration],
            'director_facebook_likes': [director_facebook_likes],
            'actor_3_facebook_likes': [actor_3_facebook_likes],
            'actor_2_facebook_likes': [actor_2_facebook_likes],
            'gross': [gross],
            'genre_type1': [genre_type1],
            'num_voted_users': [num_voted_users],
            'cast_total_facebook_likes': [cast_total_facebook_likes],
            'facenumber_in_poster': [facenumber_in_poster],
            'num_user_for_reviews': [num_user_for_reviews],
            'language': [language],
            'country': [country],
            'content_rating': [content_rating],
            'budget': [budget],
            'title_year': [title_year],
            'aspect_ratio': [aspect_ratio],
            'movie_facebook_likes': [movie_facebook_likes]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Encode categorical variables
        for col in categorical_cols:
            if col in input_df.columns:
                try:
                    input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
                except:
                    # If unseen category, use most frequent
                    input_df[col] = 0
        
        # Ensure correct column order
        input_df = input_df[feature_columns]
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Get prediction label
        predicted_class = target_encoder.inverse_transform([prediction])[0]
        
        # Create confidence scores
        class_names = target_encoder.classes_
        confidence_dict = {class_names[i]: float(prediction_proba[i]) * 100 
                          for i in range(len(class_names))}
        
        # Create result message
        result = f"""
        ## 🎬 Prediction Result
        
        ### Predicted Category: **{predicted_class}** 🎯
        
        ### Confidence Scores:
        """
        
        for class_name, conf in sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True):
            emoji = "🏆" if class_name == "Hit" else "⭐" if class_name == "Average" else "📉"
            result += f"\n{emoji} **{class_name}**: {conf:.2f}%"
        
        return result
        
    except Exception as e:
        return f"❌ Error making prediction: {str(e)}"

# Create Gradio interface
def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(theme=gr.themes.Soft(), title="Movie Success Predictor") as interface:
        
        gr.Markdown(
            """
            # 🎬 Movie Success Prediction System
            ### Predict whether your movie will be a Hit, Average, or Flop!
            
            Fill in the movie details below to get a prediction based on machine learning analysis.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Movie Statistics")
                num_critic_for_reviews = gr.Number(
                    label="Number of Critic Reviews",
                    value=200,
                    info="Total critic reviews"
                )
                num_user_for_reviews = gr.Number(
                    label="Number of User Reviews",
                    value=500,
                    info="Total user reviews"
                )
                num_voted_users = gr.Number(
                    label="Number of Voted Users",
                    value=50000,
                    info="Users who voted"
                )
                duration = gr.Number(
                    label="Duration (minutes)",
                    value=120,
                    info="Movie length"
                )
                
            with gr.Column(scale=1):
                gr.Markdown("### 💰 Financial Details")
                budget = gr.Number(
                    label="Budget (USD)",
                    value=50000000,
                    info="Production budget"
                )
                gross = gr.Number(
                    label="Gross Revenue (USD)",
                    value=100000000,
                    info="Box office revenue"
                )
                
                gr.Markdown("### 📱 Social Media Metrics")
                director_facebook_likes = gr.Number(
                    label="Director Facebook Likes",
                    value=5000
                )
                actor_2_facebook_likes = gr.Number(
                    label="Lead Actor Facebook Likes",
                    value=10000
                )
                actor_3_facebook_likes = gr.Number(
                    label="Supporting Actor Facebook Likes",
                    value=5000
                )
                
            with gr.Column(scale=1):
                gr.Markdown("### 🎭 Movie Information")
                
                cast_total_facebook_likes = gr.Number(
                    label="Total Cast Facebook Likes",
                    value=20000
                )
                movie_facebook_likes = gr.Number(
                    label="Movie Facebook Likes",
                    value=30000
                )
                facenumber_in_poster = gr.Number(
                    label="Faces in Poster",
                    value=1,
                    info="Number of faces"
                )
                
                genre_type1 = gr.Dropdown(
                    choices=['Action', 'Comedy', 'Drama', 'Thriller', 'Romance', 
                            'Horror', 'Sci-Fi', 'Adventure', 'Crime', 'Fantasy'],
                    label="Primary Genre",
                    value="Action"
                )
                language = gr.Dropdown(
                    choices=['English', 'Spanish', 'French', 'German', 'Mandarin', 'Hindi'],
                    label="Language",
                    value="English"
                )
                country = gr.Dropdown(
                    choices=['USA', 'UK', 'India', 'France', 'Germany', 'Canada'],
                    label="Country",
                    value="USA"
                )
                content_rating = gr.Dropdown(
                    choices=['G', 'PG', 'PG-13', 'R', 'NC-17', 'Not Rated'],
                    label="Content Rating",
                    value="PG-13"
                )
                title_year = gr.Number(
                    label="Release Year",
                    value=2020
                )
                aspect_ratio = gr.Number(
                    label="Aspect Ratio",
                    value=2.35,
                    info="e.g., 2.35, 1.85"
                )
        
        with gr.Row():
            predict_btn = gr.Button("🎯 Predict Movie Success", variant="primary", size="lg")
        
        with gr.Row():
            output = gr.Markdown(label="Prediction Result")
        
        predict_btn.click(
            fn=predict_movie_success,
            inputs=[
                num_critic_for_reviews, duration, director_facebook_likes,
                actor_3_facebook_likes, actor_2_facebook_likes, gross, genre_type1,
                num_voted_users, cast_total_facebook_likes, facenumber_in_poster,
                num_user_for_reviews, language, country, content_rating,
                budget, title_year, aspect_ratio, movie_facebook_likes
            ],
            outputs=output
        )
        
        gr.Markdown(
            """
            ---
            ### 📝 Notes:
            - **Hit**: IMDb score 6-10 (Highly successful)
            - **Average**: IMDb score 3-6 (Moderate success)
            - **Flop**: IMDb score 1-3 (Poor performance)
            
            *Model trained using Random Forest Classification with optimized hyperparameters*
            """
        )
    
    return interface

if __name__ == "__main__":
    # Train model first
    print("Initializing Movie Success Prediction System...")
    print("Training model... (this may take a few minutes)")
    
    try:
        # Replace with your actual dataset path
        model, scaler, feature_columns, target_encoder = train_model_pipeline("movie_metadata_master.csv")
        print("✓ Model trained successfully!")
        
        # Launch interface
        interface = create_interface()
        interface.launch(share=True)
        
    except FileNotFoundError:
        print("❌ Error: movie_metadata_master.csv not found!")
        print("Please ensure the dataset is in the same directory as this script.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")