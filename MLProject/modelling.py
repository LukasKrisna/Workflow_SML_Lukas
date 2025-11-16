import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report)
import warnings
warnings.filterwarnings('ignore')

def load_preprocessed_data():
    X_train = pd.read_csv('diabetes_preprocessing/X_train.csv')
    X_val = pd.read_csv('diabetes_preprocessing/X_val.csv')
    y_train = pd.read_csv('diabetes_preprocessing/y_train.csv').values.ravel()
    y_val = pd.read_csv('diabetes_preprocessing/y_val.csv').values.ravel()
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    return X_train, X_val, y_train, y_val

def train_basic_model(X_train, y_train, X_val, y_val):
    mlflow.set_experiment("Diabetes_Classification_Basic")
    
    mlflow.autolog()
    
    with mlflow.start_run(run_name="RandomForest_Basic_No_Tuning"):
        print("\nTraining RandomForestClassifier (no hyperparameter tuning)...")
        
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred)
        val_recall = recall_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        
        print(f"Training Accuracy:   {train_accuracy:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation Precision: {val_precision:.4f}")
        print(f"Validation Recall:    {val_recall:.4f}")
        print(f"Validation F1-Score:  {val_f1:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_val, y_val_pred))
        
        run = mlflow.active_run()
        print(f"\nMLflow Run ID: {run.info.run_id}")
        print(f"Artifact URI: {run.info.artifact_uri}")
        
    return model

def main():
    X_train, X_val, y_train, y_val = load_preprocessed_data()
    
    model = train_basic_model(X_train, y_train, X_val, y_val)

if __name__ == "__main__":
    main()