from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User

# This tells Django: "Hey, show the User model in the Admin panel!"
admin.site.register(User, UserAdmin)