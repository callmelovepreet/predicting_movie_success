# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def load_and_explore_data(filepath):
        # Load dataset
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
    # Create a copy
    db_processed = db.copy()
    
    # Drop unnecessary columns
    columns_to_drop = [
        'movie_title', 'director_name', 'actor_1_name', 'actor_2_name',
        'actor_3_name', 'plot_keywords', 'movie_imdb_link','actor_1_facebook_likes','imdb_score'
    ]
    db_processed = db_processed.drop(columns=columns_to_drop, errors='ignore')
    
    # Separate features and target
    X = db_processed.drop('Classify', axis=1)
    y = db_processed['Classify']
    
    # Identify column types
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Handle missing values
    num_imputer = SimpleImputer(strategy='median')
    X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
    
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    return X, y_encoded, target_encoder

def prepare_train_test_split(X, y):
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Apply scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train_scaled, X_test_scaled, y_train, y_test):
    rfc_model = RandomForestClassifier(random_state=42)
    rfc_model.fit(X_train_scaled, y_train)
    rfc_model_y_pred = rfc_model.predict(X_test_scaled) # Making predictions
    return rfc_model_y_pred

def tune_random_forest(X_train_scaled, X_test_scaled, y_train, y_test):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    print("\nPerforming Grid Search (this may take several minutes)...")
    
    rf_grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    rf_grid_search.fit(X_train_scaled, y_train)
    print(rf_grid_search.best_params_)
    tuned_rf_pred = rf_grid_search.best_estimator_.predict(X_test_scaled)
    tuned_rf_accuracy = accuracy_score(y_test, tuned_rf_pred) 
    return rf_grid_search.best_estimator_, tuned_rf_pred

def main():
    # Step 1: Load data
    filepath = "movie_metadata_master.csv"   # replace with your actual dataset path
    db = load_and_explore_data(filepath)

    # Step 2: Create target variable
    db = create_target_variable(db)

    # Step 3: Preprocess data
    X, y_encoded, target_encoder = preprocess_data(db)

    # Step 4: Train-test split
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_train_test_split(X, y_encoded)

    # Step 5: Train baseline Random Forest
    y_pred = train_model(X_train_scaled, X_test_scaled, y_train, y_test)

    # Step 6: Evaluate baseline model
    print("Baseline Random Forest Results:")
    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Step 7: Tune Random Forest
    best_model, tuned_pred = tune_random_forest(X_train_scaled, X_test_scaled, y_train, y_test)

    # Step 8: Evaluate tuned model
    print("\nTuned Random Forest Results:")
    print(classification_report(y_test, tuned_pred, target_names=target_encoder.classes_))
    print("Confusion Matrix:\n", confusion_matrix(y_test, tuned_pred))
    
if __name__ == "__main__":
    main()
