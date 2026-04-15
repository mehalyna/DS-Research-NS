from django.urls import path
from . import views

urlpatterns = [
    path('predict/state/', views.predict_state, name='predict_state'),
    path('cluster/', views.get_coffee_persona, name='predict_cluster'),
    path('explain/<uuid:prediction_id>/', views.explain_prediction, name='explain_prediction'),
    path('recommendation/', views.recommendation_view, name='recommendation'),
]