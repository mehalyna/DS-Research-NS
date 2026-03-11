import os
import joblib
import json
import uuid
import numpy as np
import pandas as pd
from datetime import datetime

class CoffeeHealthPredictor:
    def __init__(self, models_dir=None):
        """
        Loads the preprocessing pipeline, target encoder, and the three LightGBM models.
        """
        # Navigate up from backend/src/predictions to the main DS-RESEARCH-NS folder
        if models_dir is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            self.models_dir = os.path.join(project_root, 'models')
        else:
            self.models_dir = models_dir
            
        # Load Preprocessor
        self.preprocessor = joblib.load(os.path.join(self.models_dir, 'preprocessor.joblib'))
        
        # Load the 3 LightGBM Baseline Models
        self.model_sleep = joblib.load(os.path.join(self.models_dir, 'baselines/lightgbm_Sleep_Quality_Num.joblib'))
        self.model_stress = joblib.load(os.path.join(self.models_dir, 'baselines/lightgbm_Stress_Level_Num.joblib'))
        self.model_health = joblib.load(os.path.join(self.models_dir, 'baselines/lightgbm_Health_Issues_Num.joblib'))
        
        # Define target mapping for human-readable output
        self.sleep_map = {0: 'Poor', 1: 'Fair', 2: 'Good', 3: 'Excellent'}
        self.stress_map = {0: 'Low', 1: 'Medium', 2: 'High'}
        self.health_map = {0: 'None', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}

    def _engineer_features(self, df):
        """Applies the exact same feature engineering from Week 4."""
        df_eng = df.copy()
        
        # 1. Age Buckets
        bins = [0, 25, 35, 45, 55, 65, 100]
        labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        df_eng['Age_Group'] = pd.cut(df_eng['Age'], bins=bins, labels=labels)
        
        # 2. Coffee Strength
        df_eng['Caffeine_per_Cup'] = np.where(
            df_eng['Coffee_Intake'] > 0, 
            df_eng['Caffeine_mg'] / df_eng['Coffee_Intake'], 
            0
        )
        return df_eng

    def predict(self, user_data: dict):
        """
        Takes a dictionary of raw user data, preprocesses it, and returns predictions.
        """
        # 1. Convert dictionary to DataFrame (single row)
        df_raw = pd.DataFrame([user_data])
        
        # 2. Feature Engineering
        df_featured = self._engineer_features(df_raw)
        
        # 3. Preprocessing (Scaling & Encoding)
        # preprocessor.transform expects a DataFrame with the exact columns we used in training
        X_processed = self.preprocessor.transform(df_featured)
        
        # 4. Make Predictions & Get Probabilities
        sleep_pred = self.model_sleep.predict(X_processed)[0]
        stress_pred = self.model_stress.predict(X_processed)[0]
        health_pred = self.model_health.predict(X_processed)[0]
        
        sleep_prob = np.max(self.model_sleep.predict_proba(X_processed)[0])
        stress_prob = np.max(self.model_stress.predict_proba(X_processed)[0])
        health_prob = np.max(self.model_health.predict_proba(X_processed)[0])
        
        # 5. Format the Output
        prediction_id = str(uuid.uuid4())
        
        return {
            "prediction_id": prediction_id,
            "timestamp": datetime.now().isoformat(),
            "predictions": {
                "sleep_quality": {
                    "class": self.sleep_map[int(sleep_pred)],
                    "confidence": round(float(sleep_prob), 4)
                },
                "stress_level": {
                    "class": self.stress_map[int(stress_pred)],
                    "confidence": round(float(stress_prob), 4)
                },
                "health_issues": {
                    "class": self.health_map[int(health_pred)],
                    "confidence": round(float(health_prob), 4)
                }
            }
        }