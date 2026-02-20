from django.contrib import admin
from django.urls import path, include
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

    
urlpatterns = [
    path('admin/', admin.site.urls),
    
    # --- Authentication Endpoints ---
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/auth/', include('rest_framework.urls')), # Optional: Browsable API login

    path('api/profiles/', include('profiles.urls')), 
    path('api/records/', include('records.urls')),         
    path('api/predictions/', include('predictions.urls')), 
]