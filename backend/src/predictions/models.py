import uuid
from django.db import models

class PredictionRecord(models.Model):
    # Use a UUID instead of a standard 1, 2, 3 ID for security
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Store the raw input data from the user
    user_data = models.JSONField(help_text="Raw state data provided by the user")
    
    # Store the exact output we gave them
    predictions = models.JSONField(help_text="Model output classes and confidences")
    
    # Automatically record exactly when this happened
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction {self.id} at {self.timestamp}"