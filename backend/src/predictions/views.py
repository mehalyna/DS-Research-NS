import joblib
import os
import logging
import pandas as pd
from django.conf import settings
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.throttling import AnonRateThrottle

from utils.feature_engineering import add_derived_features
from .serializers import ClusterInputSerializer
from .predictor import CoffeeHealthPredictor
from .models import PredictionRecord

logger = logging.getLogger(__name__)

class ClusterThrottle(AnonRateThrottle):
    rate = '50/hour'

# Global initialization for heavy ML components
try:
    predictor = CoffeeHealthPredictor()
    scaler = joblib.load(os.path.join(settings.MODELS_DIR, 'clustering', 'cluster_scaler.joblib'))
    kmeans = joblib.load(os.path.join(settings.MODELS_DIR, 'clustering', 'kmeans_model.joblib'))
    metadata = joblib.load(os.path.join(settings.MODELS_DIR, 'clustering', 'cluster_metadata.joblib'))
    logger.info("Machine learning models initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize ML models: {e}")
    predictor = scaler = kmeans = metadata = None

@api_view(['POST'])
@permission_classes([AllowAny])
def predict_state(request):
    """Processes user data to generate health predictions and archives the result."""
    if not predictor:
        return Response({"error": "Service temporarily unavailable."}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        
    try:
        results = predictor.predict(request.data)
        
        # Save record for audit/explanation purposes
        PredictionRecord.objects.create(
            id=results['prediction_id'],
            user_data=request.data,
            predictions=results['predictions']
        )
        return Response(results, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return Response({"error": "Prediction processing failed."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([AllowAny])
@throttle_classes([ClusterThrottle])
def get_coffee_persona(request):
    """Categorizes user habits into predefined coffee personas via K-Means."""
    if not (scaler and kmeans):
        return Response({"error": "Clustering service unavailable."}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    
    serializer = ClusterInputSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        # Prepare data with derived features before scaling
        user_data = add_derived_features(serializer.validated_data)
        features = ['Age', 'Coffee_Intake', 'Caffeine_mg', 'Sleep_Hours', 'BMI', 'Heart_Rate', 'Physical_Activity_Hours', 'Caffeine_per_Cup']
        
        input_df = pd.DataFrame([[user_data.get(f, 0) for f in features]], columns=features)
        
        # Predict cluster based on scaled input
        cluster_id = int(kmeans.predict(scaler.transform(input_df))[0])
        
        profiles = metadata.get('profiles', {}) if metadata else {}
        return Response({
            "cluster_id": cluster_id,
            "profile": profiles.get(cluster_id, {"name": "Unknown", "description": "N/A"})
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Clustering error: {e}")
        return Response({"error": "Clustering analysis failed."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([AllowAny])
def explain_prediction(request, prediction_id):
    """Fetches original input and generates model explanations (SHAP)."""
    record = get_object_or_404(PredictionRecord, id=prediction_id)
    
    if not predictor:
        return Response({"error": "Explainer service unavailable."}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

    try:
        explanations = {
            "sleep_quality": predictor.generate_explanation('Sleep_Quality', record.user_data),
            "stress_level": predictor.generate_explanation('Stress_Level', record.user_data),
            "health_issues": predictor.generate_explanation('Health_Issues', record.user_data)
        }
        return Response({"prediction_id": prediction_id, "explanations": explanations}, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        return Response({"error": "Failed to generate explanations."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([AllowAny])
def recommendation_view(request):
    """Provides actionable advice based on current consumption profile."""
    try:
        predictor = CoffeeHealthPredictor()
        result = predictor.get_recommendation(request.data)
        
        if result['delta'] < 0:
            msg = f"We suggest reducing intake by {abs(result['delta'])} cups."
        elif result['delta'] > 0:
            msg = f"You can safely increase intake by {result['delta']} cups."
        else:
            msg = "Your intake is balanced for your health profile."
            
        return Response({"recommendation": result, "reasoning": msg})
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([AllowAny])
def check_anomaly_view(request):
    """Runs a health pattern check to identify statistical anomalies."""
    try:
        is_anomaly = CoffeeHealthPredictor().detect_anomaly(request.data)
        return Response({
            "is_anomaly": is_anomaly,
            "message": "High-risk pattern detected" if is_anomaly else "Normal pattern"
        })
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)