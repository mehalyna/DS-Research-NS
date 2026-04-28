from rest_framework import serializers
from django.contrib.auth.password_validation import validate_password
from .models import User

class UserRegistrationSerializer(serializers.ModelSerializer):
    """
    Handles user registration by validating password consistency 
    and creating a new user instance.
    """
    password = serializers.CharField(write_only=True, validators=[validate_password])
    password_confirm = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ('username', 'email', 'password', 'password_confirm')

    def validate(self, attrs):
        # Ensure password fields match before proceeding
        if attrs['password'] != attrs['password_confirm']:
            raise serializers.ValidationError({"password": "Password fields do not match."})
        return attrs

    def create(self, validated_data):
        # Exclude the confirmation field from the user object creation
        validated_data.pop('password_confirm')
        
        user = User.objects.create_user(
            username=validated_data['username'],
            email=validated_data['email'],
            password=validated_data['password']
        )
        return user