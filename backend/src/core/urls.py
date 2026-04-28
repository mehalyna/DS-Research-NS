from django.contrib import admin
from django.urls import path, include
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

"""
Core URL configuration for the Coffee Management API.
Orchestrates routing for prediction services, user authentication, and system records.
"""

urlpatterns = [
    # Admin interface for system management
    path('admin/', admin.site.urls),

    # Prediction and persona analysis services
    path('api/', include('predictions.urls')),
    
    # JWT Authentication endpoints
    path('api/auth/login/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/auth/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/auth/', include('profiles.urls')),

    # Historical data records and user activity logs
    path('api/records/', include('records.urls')),         
]