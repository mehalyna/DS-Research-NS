from django.urls import path
from . import views

urlpatterns = [
    path('predict/state/', views.predict_state, name='predict_state'),
]