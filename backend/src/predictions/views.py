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