import os
import joblib
import json
import logging
import uuid
import shap
import numpy as np
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        self.sleep_map = {0: 'Poor', 1: 'Fair', 2: 'Good', 3: 'Excellent'}
        self.stress_map = {0: 'Low', 1: 'Medium', 2: 'High'}
        self.health_map = {0: 'None', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
        
        self.model_sleep = joblib.load(os.path.join(refined_path, 'lgbm_refined_Sleep_Quality_Num.joblib'))
        self.model_stress = joblib.load(os.path.join(refined_path, 'lgbm_refined_Stress_Level_Num.joblib'))
        self.model_health = joblib.load(os.path.join(refined_path, 'lgbm_refined_Health_Issues_Num.joblib'))

        self.models = {
            'Sleep_Quality': self.model_sleep,
            'Stress_Level': self.model_stress,
            'Health_Issues': self.model_health
        }

        # Load Anomaly Detector
        anomaly_path = os.path.join(self.models_dir, 'anomaly', 'iso_forest_v1.joblib')
        self.anomaly_detector = joblib.load(anomaly_path)
        
        # Features used during training
        self.risk_features = ['Age', 'BMI', 'Heart_Rate', 'Coffee_Intake', 'Sleep_Hours']
    
    def _log_event(self, event_type, data, result):
        """Internal helper to track system behavior for the integration pass."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "input_cups": data.get('Coffee_Intake'),
            "is_anomaly": result.get('is_anomaly', False) if isinstance(result, dict) else "N/A",
            "recommendation": result.get('recommended_cups', "N/A") if isinstance(result, dict) else "N/A"
        }
        logger.info(f"COFFEE_AI_LOG: {log_entry}")

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
        
        output = {
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

        self._log_event("prediction_made", user_data, output)
        return output
    
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

        translation_map = {
            "num__Age": "As we age, the liver processes caffeine more slowly, which can extend its stimulating effects into the night.",
            "num__Heart_Rate": "A higher resting heart rate indicates your cardiovascular system is already under load; caffeine acts as an additional stimulant.",
            "num__Sleep_Hours": "Quality sleep is the only way for the brain to recover; caffeine can mask exhaustion but cannot replace restorative rest.",
            "num__BMI": "Body composition affects how caffeine is distributed and metabolized throughout your system.",
            "num__Coffee_Intake": "Your current daily consumption sets the baseline for your body's caffeine tolerance and dependency.",
            "num__Physical_Activity_Hours": "Exercise provides a 'stress buffer' that helps your body process caffeine more effectively without the jitters.",
            "cat__Smoking_Yes": "Smoking can accelerate caffeine metabolism, often leading to a cycle of higher consumption and lower sleep quality.",
            "cat__Occupation_Student": "Students often face irregular schedules, making them more susceptible to caffeine-induced sleep disruption."
        }

        # Format into JSON-ready dictionary 
        feature_impacts = []
        for i, feature_name in enumerate(user_features_df.columns):
            shap_val = float(user_shap_values[i])
            
            if abs(shap_val) < 0.001:
                continue
            
            # Use the map to get a human explanation, or fallback to generic
            explanation = translation_map.get(feature_name, f"This factor is a key driver for your {target_model.replace('_', ' ')}.")
                
            feature_impacts.append({
                "feature": feature_name.split('__')[-1].replace('_', ' '),
                "shap_influence": round(shap_val, 4),
                "impact": "positive" if shap_val > 0 else "negative",
                "display_text": explanation # This is what your Streamlit UI will now show
            })

        feature_impacts.sort(key=lambda x: abs(x["shap_influence"]), reverse=True)
        top_features = feature_impacts[:5]

        return {
            "target_model": target_model.replace('_', ' '),
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

        sleep_probs = self.models['Sleep_Quality'].predict_proba(X_processed)
        stress_probs = self.models['Stress_Level'].predict_proba(X_processed)
        
        # 4. Scoring Logic (Matching your Week 13 Notebook)
        utility_bonus = df_scenarios['Coffee_Intake'] * 0.1
        scores = (sleep_preds - stress_preds - health_preds) + utility_bonus
        
        # 5. Safety Guardrails (Rule-based Fallback)
        def is_safe(row):
            # 1. Tachycardia & Absolute FDA Cap
            if user_data['Heart_Rate'] > 105 and row['Coffee_Intake'] > 0.5:
                return False
            if row['Caffeine_mg'] > 400:
                return False
            
            # 2. The "Nudge" Constraint: Prevent overwhelming the user
            # Don't suggest increasing by more than 1.5 cups from current baseline
            if row['Coffee_Intake'] > (user_data['Coffee_Intake']):
                return False
            # Don't suggest decreasing by more than 2.0 cups (avoid heavy withdrawal)
            if row['Coffee_Intake'] < (user_data['Coffee_Intake'] - 2.0):
                return False

            # 3. Contextual Risk Multipliers (Smoking, Age, BMI)
            risk_factors = 0
            if user_data['Smoking'] == 'Yes': risk_factors += 1
            if user_data['BMI'] > 30: risk_factors += 1
            if user_data['Age'] > 60: risk_factors += 1
            
            # If 2+ risks are present, strictly cap at 2.0 cups regardless of model score
            if risk_factors >= 2 and row['Coffee_Intake'] > 2.0:
                return False

            # 4. Sleep Deprivation Guard
            # If current sleep is very poor, we never recommend an increase
            if user_data['Sleep_Hours'] < 5.5 and row['Coffee_Intake'] > user_data['Coffee_Intake']:
                return False

            return True
        
        df_scenarios['is_safe'] = df_scenarios.apply(is_safe, axis=1)
        safe_df = df_scenarios[df_scenarios['is_safe']]
        
        # If no scenario is safe, provide a strict emergency recommendation
        if safe_df.empty:
            return {
                "recommended_cups": 0.0, 
                "delta": round(float(-user_data['Coffee_Intake']), 1), 
                "impact_sleep": "Critical",
                "impact_stress": "High Risk",
                "stress_reduction_pct": 0.0,
                "reasoning": "Your current physiological markers (Sleep/Heart Rate) indicate extreme strain. No additional caffeine is safely recommended."
            }
        
        # Calculate indices for safe scenarios to match probability arrays
        safe_indices = safe_df.index.tolist()
        
        # New Scoring: Utility is secondary to stress and sleep
        # sleep_probs columns index: 2 is 'Good', 3 is 'Excellent'
        # stress_probs column index: 2 is 'High'
        sleep_score = sleep_probs[safe_indices, 2] + sleep_probs[safe_indices, 3]
        stress_penalty = stress_probs[safe_indices, 2]
        
        scores = sleep_score - stress_penalty + (cup_options[safe_indices] * 0.1)
        
        best_relative_idx = np.argmax(scores)
        best_safe_idx = safe_indices[best_relative_idx]
        best_cups = cup_options[best_safe_idx]

        # Calculate "Impact Probability" - Risk reduction of High Stress
        current_high_stress_prob = stress_probs[0, 2]
        new_high_stress_prob = stress_probs[best_safe_idx, 2]
        prob_reduction = (current_high_stress_prob - new_high_stress_prob) / (current_high_stress_prob + 1e-6)
        
        output = {
            "recommended_cups": float(best_cups),
            "original_intake": float(user_data['Coffee_Intake']),
            "delta": round(float(best_cups - user_data['Coffee_Intake']), 1),
            "impact_sleep": "Improvement" if sleep_preds[best_safe_idx] > sleep_preds[0] else "Stable",
            "impact_stress": "Reduction" if stress_preds[best_safe_idx] < stress_preds[0] else "Stable",
            "stress_reduction_pct": round(max(0, prob_reduction * 100), 1),
            "reasoning": "This recommendation balances your cardiovascular safety with metabolic data to find a stable intake level."
        }

        self._log_event("recommendation_generated", user_data, output)
        return output
    
    def detect_anomaly(self, user_data: dict):
        """Returns True if the input pattern is a risky outlier."""
        # Create DF with only the 5 risk features
        df_risk = pd.DataFrame([user_data])[self.risk_features]
        
        # Predict returns 1 (normal) or -1 (anomaly)
        prediction = self.anomaly_detector.predict(df_risk)[0]
        return True if prediction == -1 else False