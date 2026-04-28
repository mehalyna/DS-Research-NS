from rest_framework import serializers

class ClusterInputSerializer(serializers.Serializer):
    """
    Validates input features required for the caffeine persona clustering model.
    """
    Age = serializers.IntegerField(min_value=1, max_value=200)
    Coffee_Intake = serializers.FloatField(min_value=0, max_value=20)
    Caffeine_mg = serializers.FloatField(min_value=0, max_value=5000)
    Sleep_Hours = serializers.FloatField(min_value=0, max_value=24)
    BMI = serializers.FloatField(min_value=10, max_value=50)
    Heart_Rate = serializers.IntegerField(min_value=30, max_value=200)
    Physical_Activity_Hours = serializers.FloatField(min_value=0, max_value=150)