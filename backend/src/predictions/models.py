import uuid
from django.db import models

class PredictionRecord(models.Model):
    """
    Stores historical prediction data including user inputs,
    model outputs, and the time of prediction.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_data = models.JSONField(help_text="Raw state data provided by the user")
    predictions = models.JSONField(help_text="Model output classes and confidence scores")
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction {self.id} on {self.timestamp.strftime('%Y-%m-%d %H:%M')}"