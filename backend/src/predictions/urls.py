from django.urls import path
from .views import predict_state, get_coffee_persona

urlpatterns = [
    path('predict/state/', predict_state, name='predict_state'),
    path('cluster/', get_coffee_persona, name='predict_cluster')
]