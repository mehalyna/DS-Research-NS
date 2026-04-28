import os
import joblib
import logging
import uuid
import shap
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class CoffeeHealthPredictor:
    """
    Handles model inference, feature engineering, and anomaly detection 
    for the caffeine management system.
    """
    def __init__(self, models_dir=None):
        # Determine the project root for consistent pathing
        if models_dir is None:
            self.models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models'))
        else:
            self.models_dir = models_dir
            
        # Load preprocessor and anomaly detector
        try:
            self.preprocessor = joblib.load(os.path.join(self.models_dir, 'preprocessor.joblib'))
            self.anomaly_detector = joblib.load(os.path.join(self.models_dir, 'anomaly', 'iso_forest_v1.joblib'))
        except Exception as e:
            logger.error(f"Failed to load infrastructure models: {e}")
            raise

        # Initialize target mappings for categorical labels
        self.mappings = {
            'Sleep_Quality': {0: 'Poor', 1: 'Fair', 2: 'Good', 3: 'Excellent'},
            'Stress_Level': {0: 'Low', 1: 'Medium', 2: 'High'},
            'Health_Issues': {0: 'None', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
        }
        
        # Load prediction models
        refined_path = os.path.join(self.models_dir, 'refined')
        self.models = {
            'Sleep_Quality': joblib.load(os.path.join(refined_path, 'lgbm_refined_Sleep_Quality_Num.joblib')),
            'Stress_Level': joblib.load(os.path.join(refined_path, 'lgbm_refined_Stress_Level_Num.joblib')),
            'Health_Issues': joblib.load(os.path.join(refined_path, 'lgbm_refined_Health_Issues_Num.joblib'))
        }
        
        self.risk_features = ['Age', 'BMI', 'Heart_Rate', 'Coffee_Intake', 'Sleep_Hours']
    
    def _log_event(self, event_type, data, result):
        """Records system activity for monitoring and debugging purposes."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "input_cups": data.get('Coffee_Intake'),
            "is_anomaly": result.get('is_anomaly', False) if isinstance(result, dict) else "N/A"
        }
        logger.info(f"System event logged: {log_entry}")

    def _engineer_features(self, df):
        """Applies feature engineering pipeline to match training data transformations."""
        df_eng = df.copy()
        
        # Define age brackets and calculate caffeine density
        bins = [0, 25, 35, 45, 55, 65, 100]
        labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        df_eng['Age_Group'] = pd.cut(df_eng['Age'], bins=bins, labels=labels)
        
        df_eng['Caffeine_per_Cup'] = np.where(
            df_eng['Coffee_Intake'] > 0, 
            df_eng['Caffeine_mg'] / df_eng['Coffee_Intake'], 
            0
        )
        return df_eng

    def predict(self, user_data: dict):
        """
        Processes raw user data, runs it through the ML pipeline, 
        and returns structured predictions with confidence scores.
        """
        # Prepare input data
        df_raw = pd.DataFrame([user_data])
        df_featured = self._engineer_features(df_raw)
        X_processed = self.preprocessor.transform(df_featured)
        
        # Collect predictions and probabilities
        output = {
            "prediction_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "predictions": {}
        }
        
        for name, model in self.models.items():
            pred = model.predict(X_processed)[0]
            prob = np.max(model.predict_proba(X_processed)[0])
            
            output["predictions"][name.lower()] = {
                "class": self.mappings[name][int(pred)],
                "confidence": round(float(prob), 4)
            }
        
        self._log_event("prediction_made", user_data, output)
        return output
    
    def generate_explanation(self, target_model: str, user_data: dict):
        """
        Generates SHAP-based feature importance explanations for a prediction.
        Maps raw input features to clinical interpretability text.
        """
        calibrated_wrapper = self.models.get(target_model)
        if not calibrated_wrapper:
             raise ValueError(f"Model {target_model} not found.")
        
        # Access the underlying estimator within the calibration wrapper
        actual_model = calibrated_wrapper.calibrated_classifiers_[0].estimator
        explainer = shap.TreeExplainer(actual_model)

        # Preprocess input data to match expected feature space
        df_raw = pd.DataFrame([user_data])
        df_featured = self._engineer_features(df_raw)
        X_processed = self.preprocessor.transform(df_featured)
        
        if hasattr(X_processed, 'toarray'):
            X_processed = X_processed.toarray()
            
        feature_names = self.preprocessor.get_feature_names_out()
        user_features_df = pd.DataFrame(X_processed, columns=feature_names)

        # Generate SHAP values
        shap_explanation = explainer(user_features_df)
        
        # Handle binary classification SHAP shapes
        if len(shap_explanation.values.shape) == 3:
            user_shap_values = shap_explanation.values[0, :, 1]
        else:
            user_shap_values = shap_explanation.values[0]

        # Clinical translation map for end-user interpretability
        explanation_map = {
            "num__Age": "As we age, the liver processes caffeine more slowly, extending stimulating effects.",
            "num__Heart_Rate": "Higher heart rates indicate cardiovascular load; caffeine acts as an additional stimulant.",
            "num__Sleep_Hours": "Caffeine masks exhaustion but cannot replace the restorative power of sleep.",
            "num__BMI": "Body composition influences how caffeine is distributed and metabolized.",
            "num__Coffee_Intake": "Your current daily consumption sets your baseline tolerance and dependency.",
            "num__Physical_Activity_Hours": "Exercise acts as a stress buffer, helping process caffeine more effectively.",
            "cat__Smoking_Yes": "Smoking accelerates caffeine metabolism, often triggering cycles of higher intake.",
            "cat__Occupation_Student": "Irregular schedules increase susceptibility to caffeine-induced sleep disruption."
        }

        # Calculate feature impact rankings
        impacts = []
        for i, name in enumerate(user_features_df.columns):
            shap_val = float(user_shap_values[i])
            if abs(shap_val) < 0.001:
                continue
            
            clean_name = name.split('__')[-1].replace('_', ' ')
            impacts.append({
                "feature": clean_name,
                "shap_influence": round(shap_val, 4),
                "impact": "positive" if shap_val > 0 else "negative",
                "display_text": explanation_map.get(name, f"Key driver for {target_model.replace('_', ' ')}.")
            })

        impacts.sort(key=lambda x: abs(x["shap_influence"]), reverse=True)
        
        return {
            "target_model": target_model.replace('_', ' '),
            "top_drivers": impacts[:5]
        }
    
    def get_recommendation(self, user_data: dict):
        """
        Evaluates intake scenarios to provide an optimal, safe coffee recommendation 
        based on physiological health markers and rule-based guardrails.
        """
        # 1. Generate intake scenarios (0 to 5 cups in 0.5 increments)
        cup_options = np.arange(0, 5.5, 0.5)
        scenarios = [
            {**user_data, 'Coffee_Intake': c, 'Caffeine_mg': c * 95.0} 
            for c in cup_options
        ]
        df_scenarios = pd.DataFrame(scenarios)
        
        # 2. Process features and get model probabilities
        df_featured = self._engineer_features(df_scenarios)
        X_processed = self.preprocessor.transform(df_featured)
        
        sleep_preds = self.models['Sleep_Quality'].predict(X_processed)
        stress_preds = self.models['Stress_Level'].predict(X_processed)
        
        sleep_probs = self.models['Sleep_Quality'].predict_proba(X_processed)
        stress_probs = self.models['Stress_Level'].predict_proba(X_processed)
        
        # 3. Apply safety constraints
        def is_safe(row):
            # Enforce FDA caps and physiological limits
            if user_data['Heart_Rate'] > 105 and row['Coffee_Intake'] > 0.5: return False
            if row['Caffeine_mg'] > 400: return False
            
            # Limit consumption variance to prevent withdrawal or over-stimulation
            if row['Coffee_Intake'] > user_data['Coffee_Intake'] or row['Coffee_Intake'] < (user_data['Coffee_Intake'] - 2.0):
                return False

            # Contextual risk assessment
            risk_factors = sum([user_data.get('Smoking') == 'Yes', user_data.get('BMI', 0) > 30, user_data.get('Age', 0) > 60])
            if risk_factors >= 2 and row['Coffee_Intake'] > 2.0: return False
            if user_data.get('Sleep_Hours', 0) < 5.5 and row['Coffee_Intake'] > user_data['Coffee_Intake']: return False

            return True
        
        df_scenarios['is_safe'] = df_scenarios.apply(is_safe, axis=1)
        safe_df = df_scenarios[df_scenarios['is_safe']]
        
        if safe_df.empty:
            return {
                "recommended_cups": 0.0, 
                "delta": -float(user_data['Coffee_Intake']), 
                "reasoning": "Current physiological indicators suggest extreme strain; no caffeine recommended."
            }
        
        # 4. Rank safe scenarios by predicted sleep quality and stress reduction
        safe_indices = safe_df.index.tolist()
        sleep_score = sleep_probs[safe_indices, 2] + sleep_probs[safe_indices, 3]
        stress_penalty = stress_probs[safe_indices, 2]
        
        scores = sleep_score - stress_penalty + (cup_options[safe_indices] * 0.1)
        
        best_idx = safe_indices[np.argmax(scores)]
        
        # 5. Compile output
        prob_reduction = (stress_probs[0, 2] - stress_probs[best_idx, 2]) / (stress_probs[0, 2] + 1e-6)
        
        output = {
            "recommended_cups": float(cup_options[best_idx]),
            "delta": round(float(cup_options[best_idx] - user_data['Coffee_Intake']), 1),
            "impact_sleep": "Improvement" if sleep_preds[best_idx] > sleep_preds[0] else "Stable",
            "impact_stress": "Reduction" if stress_preds[best_idx] < stress_preds[0] else "Stable",
            "stress_reduction_pct": round(max(0, prob_reduction * 100), 1),
            "reasoning": "Recommendation balances cardiovascular markers with metabolic health."
        }

        self._log_event("recommendation_generated", user_data, output)
        return output
    
    def detect_anomaly(self, user_data: dict):
        """
        Evaluates input features against an Isolation Forest model to flag 
        statistically significant physiological outliers.
        """
        df_risk = pd.DataFrame([user_data])[self.risk_features]
        return bool(self.anomaly_detector.predict(df_risk)[0] == -1)