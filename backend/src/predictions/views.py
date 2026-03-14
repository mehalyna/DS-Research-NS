import joblib
import os
from django.conf import settings
import pandas as pd
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from .predictor import CoffeeHealthPredictor
from .models import PredictionRecord

# Initialize the predictor once when the server starts
# (This prevents loading the heavy models every single time a request comes in)
try:
    predictor = CoffeeHealthPredictor()
    print("✓ ML Predictor loaded successfully into Django.")
except Exception as e:
    print(f"❌ Failed to load ML Predictor: {e}")
    predictor = None

@api_view(['POST'])
@permission_classes([AllowAny])
def predict_state(request):
    """
    Takes user state data, runs it through the ML pipeline, and returns predictions.
    """
    if not predictor:
        return Response(
            {"error": "Machine learning model is currently unavailable."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )
        
    try:
        # request.data contains the JSON payload sent by the user
        user_data = request.data
        
        # Run inference
        results = predictor.predict(user_data)
        
        PredictionRecord.objects.create(
            id=results['prediction_id'],
            user_data=user_data,
            predictions=results['predictions']
        )
        
        # Return the results as a clean JSON response
        return Response(results, status=status.HTTP_200_OK)
        
    except KeyError as e:
        return Response(
            {"error": f"Missing required data field: {str(e)}" },
            status=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        return Response(
            {"error": f"Prediction failed: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
# Define paths to the models you just saved
SCALER_PATH = os.path.join(settings.BASE_DIR, '../../models/clustering/cluster_scaler.joblib')
KMEANS_PATH = os.path.join(settings.BASE_DIR, '../../models/clustering/kmeans_model.joblib')

# Load them globally so they don't reload on every single request
cluster_scaler = joblib.load(SCALER_PATH)
kmeans_model = joblib.load(KMEANS_PATH)

@api_view(['POST'])
@permission_classes([AllowAny])
def predict_cluster(request):
    """
    Takes user state data and returns their Coffee Persona (Cluster).
    """
    user_data = request.data
    
    # The exact 8 features we used in our Week 8 notebook, in the exact same order
    cluster_features = [
        'Age', 'Coffee_Intake', 'Caffeine_mg', 'Sleep_Hours', 
        'BMI', 'Heart_Rate', 'Physical_Activity_Hours', 'Caffeine_per_Cup'
    ]
    
    try:
        # 1. Extract the numbers from the request
        input_values = [[user_data.get(feat, 0) for feat in cluster_features]]
        
        # 2. Convert to Pandas DataFrame to avoid scikit-learn warnings about feature names
        input_df = pd.DataFrame(input_values, columns=cluster_features)
        
        # 3. Scale the data using the exact same mathematical rules from Week 8
        scaled_data = cluster_scaler.transform(input_df)
        
        # 4. Predict the cluster (0, 1, or 2)
        cluster_id = int(kmeans_model.predict(scaled_data)[0])
        
        # 5. Add a friendly profile description based on our analysis!
        profiles = {
            0: {"name": "The High-Octane Chugger", "description": "High caffeine, low sleep, elevated heart rate."},
            1: {"name": "The Decaf Abstainer", "description": "Zero coffee, balanced sleep, low heart rate."},
            2: {"name": "The Balanced Brewer", "description": "Moderate coffee, well-rested, normal heart rate."}
        }
        
        return Response({
            "cluster_id": cluster_id,
            "profile": profiles.get(cluster_id, {"name": "Unknown", "description": "N/A"})
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)