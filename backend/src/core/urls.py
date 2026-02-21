from django.contrib import admin
from django.urls import path, include
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # --- Authentication Endpoints ---
    path('api/auth/login/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/auth/refresh/', TokenRefreshView.as_view(), name='token_refresh'),

    path('api/auth/', include('profiles.urls')),

    # --- The API Skeleton ---
    path('api/records/', include('records.urls')),         
    path('api/predictions/', include('predictions.urls')), 
]