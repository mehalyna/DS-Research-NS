import os
import joblib
import json
import uuid
import shap
import numpy as np
import pandas as pd
from datetime import datetime

class CoffeeHealthPredictor:
    def __init__(self, models_dir=None):
        """
        Loads the preprocessing pipeline, target encoder, and the three LightGBM models.
        """

        if models_dir is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            self.models_dir = os.path.join(project_root, 'models')
        else:
            self.models_dir = models_dir
            
        # Load Preprocessor
        self.preprocessor = joblib.load(os.path.join(self.models_dir, 'preprocessor.joblib'))
        
        refined_path = os.path.join(self.models_dir, 'refined')
        
        self.model_sleep = joblib.load(os.path.join(refined_path, 'lgbm_refined_Sleep_Quality_Num.joblib'))
        self.model_stress = joblib.load(os.path.join(refined_path, 'lgbm_refined_Stress_Level_Num.joblib'))
        self.model_health = joblib.load(os.path.join(refined_path, 'lgbm_refined_Health_Issues_Num.joblib'))

        self.sleep_map = {0: 'Poor', 1: 'Fair', 2: 'Good', 3: 'Excellent'}
        self.stress_map = {0: 'Low', 1: 'Medium', 2: 'High'}
        self.health_map = {0: 'None', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
        
        self.models = {
            'Sleep_Quality': self.model_sleep,
            'Stress_Level': self.model_stress,
            'Health_Issues': self.model_health
        }

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
    
    def generate_explanation(self, target_model: str, user_data: dict):
        """
        Takes RAW user data, preprocesses it into the 44 expected columns, 
        generates SHAP values, and formats them into JSON.
        """
        calibrated_wrapper = self.models.get(target_model)
        
        if not calibrated_wrapper:
             raise ValueError(f"Model {target_model} not found.")
        
        actual_model = calibrated_wrapper.calibrated_classifiers_[0].estimator

        explainer = shap.TreeExplainer(actual_model)
        
        if not explainer:
            raise ValueError(f"No cached explainer found for {target_model}")

        # 1. Preprocess the raw input dictionary
        df_raw = pd.DataFrame([user_data])
        df_featured = self._engineer_features(df_raw)
        X_processed = self.preprocessor.transform(df_featured)
        
        if hasattr(X_processed, 'toarray'):
            X_processed = X_processed.toarray()
            
        feature_names = self.preprocessor.get_feature_names_out()
        user_features_df = pd.DataFrame(X_processed, columns=feature_names)

        # 2. Run SHAP
        shap_explanation = explainer(user_features_df)
        
        # 3. Extract values based on model type
        if len(shap_explanation.values.shape) == 3:
            user_shap_values = shap_explanation.values[0, :, 1]
            base_value = float(shap_explanation.base_values[0, 1])
        else:
            user_shap_values = shap_explanation.values[0]
            base_value = float(shap_explanation.base_values[0])
            
        user_data_array = shap_explanation.data[0]

        # 4. Format into JSON-ready dictionary
        feature_impacts = []
        all_shap_sum = 0.0

        for i, feature_name in enumerate(user_features_df.columns):
            shap_val = float(user_shap_values[i])
            feat_val = float(user_data_array[i])
            all_shap_sum += shap_val
            
            if abs(shap_val) < 0.001:
                continue
                
            feature_impacts.append({
                "feature": feature_name,
                "value": round(feat_val, 2),
                "shap_influence": round(shap_val, 4)
            })

        feature_impacts.sort(key=lambda x: abs(x["shap_influence"]), reverse=True)
        top_features = feature_impacts[:5]

        for item in top_features:
            influence = item["shap_influence"]
            item["impact"] = "positive" if influence > 0 else "negative"
            direction = "increased" if influence > 0 else "decreased"
            item["display_text"] = f"Your {item['feature']} value of {item['value']} {direction} your score."

        final_prediction = base_value + all_shap_sum

        return {
            "target_model": target_model,
            "base_value": round(base_value, 4),
            "final_prediction": round(final_prediction, 4),
            "top_drivers": top_features
        }
    
    def get_recommendation(self, user_data: dict):
        """
        Simulates scenarios and returns the optimal coffee intake.
        """
        # 1. Generate Scenarios (0 to 5 cups)
        cup_options = np.arange(0, 5.5, 0.5)
        scenarios = []
        for cups in cup_options:
            s = user_data.copy()
            s['Coffee_Intake'] = cups
            s['Caffeine_mg'] = cups * 95.0
            scenarios.append(s)
        
        df_scenarios = pd.DataFrame(scenarios)
        
        # 2. Engineering & Preprocessing
        df_featured = self._engineer_features(df_scenarios)
        X_processed = self.preprocessor.transform(df_featured)
        
        # 3. Predict Outcomes using the calibrated models
        sleep_preds = self.models['Sleep_Quality'].predict(X_processed)
        stress_preds = self.models['Stress_Level'].predict(X_processed)
        health_preds = self.models['Health_Issues'].predict(X_processed)
        
        # 4. Scoring Logic (Matching your Week 13 Notebook)
        utility_bonus = df_scenarios['Coffee_Intake'] * 0.15
        scores = (sleep_preds - stress_preds - health_preds) + utility_bonus
        
        # 5. Safety Guardrails (Rule-based Fallback)
        def is_safe(row):
            # 1. Tachycardia Guard (Heart Rate > 100)
            if user_data['Heart_Rate'] > 100 and row['Coffee_Intake'] > 0.5:
                return False
            
            # 2. FDA Cap (400mg)
            if row['Caffeine_mg'] > 400:
                return False
            
            # 3. Age-based limit (Over 65 should be cautious)
            if user_data['Age'] > 65 and row['Coffee_Intake'] > 2.5:
                return False

            # 4. Sleep Deprivation Guard
            # If user sleeps < 5 hours, don't recommend increasing caffeine
            if user_data['Sleep_Hours'] < 5.0 and row['Coffee_Intake'] > user_data['Coffee_Intake']:
                return False

            return True

        df_scenarios['health_score'] = scores
        df_scenarios['is_safe'] = df_scenarios.apply(is_safe, axis=1)
        
        # 6. Find Best Safe Option
        safe_df = df_scenarios[df_scenarios['is_safe']]
        if safe_df.empty: return {"recommended_cups": 0.0, "delta": -user_data['Coffee_Intake']}
        
        best_idx = safe_df['health_score'].idxmax()
        best_cups = safe_df.loc[best_idx, 'Coffee_Intake']
        
        return {
            "recommended_cups": float(best_cups),
            "original_intake": float(user_data['Coffee_Intake']),
            "delta": float(best_cups - user_data['Coffee_Intake']),
            # Logic for "expected effects"
            "impact_sleep": "Improvement" if sleep_preds[best_idx] > sleep_preds[0] else "Stable",
            "impact_stress": "Reduction" if stress_preds[best_idx] < stress_preds[0] else "Stable"
        }