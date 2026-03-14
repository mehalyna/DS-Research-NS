from django.urls import path
from .views import predict_state, predict_cluster

urlpatterns = [
    path('predict/state/', predict_state, name='predict_state'),
    path('cluster/', predict_cluster, name='predict_cluster'),
]