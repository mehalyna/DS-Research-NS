import joblib
import os
import pandas as pd
from utils.feature_engineering import add_derived_features
from .serializers import ClusterInputSerializer
from django.conf import settings
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.throttling import AnonRateThrottle
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from .predictor import CoffeeHealthPredictor
from .models import PredictionRecord

class ClusterThrottle(AnonRateThrottle):
    rate = '50/hour'

SCALER_PATH = os.path.join(settings.MODELS_DIR, 'clustering', 'cluster_scaler.joblib')
KMEANS_PATH = os.path.join(settings.MODELS_DIR, 'clustering', 'kmeans_model.joblib')
METADATA_PATH = os.path.join(settings.MODELS_DIR, 'clustering', 'cluster_metadata.joblib')

# Initialize the predictor once when the server starts
# (This prevents loading the heavy models every single time a request comes in)
try:
    predictor = CoffeeHealthPredictor()
    print("✓ ML Predictor loaded successfully into Django.")
except Exception as e:
    print(f"❌ Failed to load ML Predictor: {e}")
    predictor = None

try:
    cluster_scaler = joblib.load(SCALER_PATH)
    kmeans_model = joblib.load(KMEANS_PATH)
    cluster_metadata = joblib.load(METADATA_PATH)
    print("✓ Clustering models loaded successfully.")
except Exception as e:
    print(f"⚠️ Clustering models failed to load: {e}")
    cluster_scaler = None
    kmeans_model = None
    cluster_metadata = None

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
    
@api_view(['POST'])
@permission_classes([AllowAny])
@throttle_classes([ClusterThrottle])
def get_coffee_persona(request):
    """
    Assign user to one of 3 Coffee Personas based on their habits.
    
    **Request Body:**
    ```json
    {
        "Age": 25,
        "Coffee_Intake": 3.0,
        "Caffeine_mg": 285.0,
        "Sleep_Hours": 7.0,
        "BMI": 22.5,
        "Heart_Rate": 70,
        "Physical_Activity_Hours": 5.0
    }
    ```
    
    **Response:** 200 OK
    ```json
    {
        "cluster_id": 2,
        "profile": {
            "name": "The Balanced Brewer",
            "description": "Moderate coffee, well-rested, normal heart rate."
        }
    }
    ```
    """

    if not cluster_scaler or not kmeans_model:
        return Response(
            {"error": "Clustering models are currently unavailable."}, 
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    
    serializer = ClusterInputSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    # Use the validated data instead of the raw request data
    validated_data = serializer.validated_data
    
    # --- Let the backend do the math! ---
    user_data = add_derived_features(validated_data)
    
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
        
        if cluster_metadata and 'profiles' in cluster_metadata:
            profile_data = cluster_metadata['profiles'].get(cluster_id, {"name": "Unknown", "description": "N/A"})
        else:
            profile_data = {"name": "Unknown", "description": "Metadata missing"}
        
        return Response({
            "cluster_id": cluster_id,
            "profile": profile_data
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)