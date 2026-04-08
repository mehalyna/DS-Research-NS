from django.urls import path
from .views import predict_state, get_coffee_persona, explain_prediction

urlpatterns = [
    path('predict/state/', predict_state, name='predict_state'),
    path('cluster/', get_coffee_persona, name='predict_cluster'),
    path('explain/<uuid:prediction_id>/', explain_prediction, name='explain_prediction')
]