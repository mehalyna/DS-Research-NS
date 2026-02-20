from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    # We inherit everything from standard Django User (username, password, email)
    # But now we can add custom fields easily!
    
    bio = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return self.username