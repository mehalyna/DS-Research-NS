from django.test import TestCase, Client
from rest_framework.test import APITestCase
from rest_framework import status
from django.urls import reverse
import json

class CoffeeAITests(TestCase):
    def setUp(self):
        self.client = Client()
        # Define a baseline risky payload
        self.risky_user = {
            "Age": 70, "BMI": 30.0, "Heart_Rate": 110, 
            "Coffee_Intake": 4.0, "Caffeine_mg": 380.0,
            "Sleep_Hours": 4.0, "Physical_Activity_Hours": 1.0,
            "Gender": "Male", "Occupation": "Office", "Country": "Germany",
            "Smoking": "No", "Alcohol_Consumption": "No"
        }

    def test_recommendation_safety_logic(self):
        """Test that age and heart rate guardrails restrict intake."""
        url = reverse('recommendation')
        response = self.client.post(url, data=json.dumps(self.risky_user), content_type='application/json')
        res_data = response.json()
        
        # Guardrail 1: Age > 65 should limit intake to 2.5
        # Guardrail 2: HR > 100 should limit intake to 0.5
        # The strictest rule (0.5) should win.
        self.assertLessEqual(res_data['recommendation']['recommended_cups'], 0.5)

    def test_anomaly_detection_endpoint(self):
        """Test that extreme data triggers the anomaly flag."""
        url = reverse('check-anomaly')
        response = self.client.post(url, data=json.dumps(self.risky_user), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['is_anomaly'])
        
class PredictionAPITests(APITestCase):
    
    def setUp(self):
        # We use 'predict_state' because that is the 'name' we gave it in predictions/urls.py
        self.url = reverse('predict_state')
        self.valid_payload = {
            "Age": 21,
            "Gender": "Female",
            "Country": "Ukraine",
            "Coffee_Intake": 4,
            "Caffeine_mg": 450,
            "Sleep_Hours": 5.5,
            "BMI": 21.5,
            "Heart_Rate": 82,
            "Physical_Activity_Hours": 2.0,
            "Occupation": "Student",
            "Smoking": "No",
            "Alcohol_Consumption": "Yes"
        }

    def test_predict_state_success(self):
        """Test that sending a valid payload returns a 200 OK and valid predictions."""
        response = self.client.post(self.url, self.valid_payload, format='json')
        
        # Check that the request was successful
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # Check that the response contains our expected keys
        self.assertIn('prediction_id', response.data)
        self.assertIn('predictions', response.data)
        self.assertIn('sleep_quality', response.data['predictions'])

    def test_predict_state_missing_data(self):
        """Test that missing data is handled safely (doesn't crash with 200)."""
        bad_payload = {"Age": 21}  # Missing almost everything
        response = self.client.post(self.url, bad_payload, format='json')
        
        # Should return a 400 Bad Request or 500 Error, but definitely not 200 OK
        self.assertNotEqual(response.status_code, status.HTTP_200_OK)
    
    def test_predict_cluster_success(self):
        """Test that sending user stats returns a valid Coffee Persona."""
        cluster_url = reverse('predict_cluster')
        cluster_payload = {
            "Age": 35,
            "Coffee_Intake": 0.0,
            "Caffeine_mg": 0.0,
            "Sleep_Hours": 8.0,
            "BMI": 22.0,
            "Heart_Rate": 60,
            "Physical_Activity_Hours": 8.0,
            "Caffeine_per_Cup": 0.0
        }
        
        response = self.client.post(cluster_url, cluster_payload, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('cluster_id', response.data)
        self.assertIn('profile', response.data)
        self.assertIn('name', response.data['profile'])
    
    def test_predict_cluster_zero_coffee(self):
        """Verify zero-coffee users get assigned to Cluster 1 (Decaf Abstainer)."""
        cluster_url = reverse('predict_cluster')
        payload = {
            "Age": 25,
            "Coffee_Intake": 0.0,
            "Caffeine_mg": 0.0,
            "Sleep_Hours": 8.0,
            "BMI": 22.0,
            "Heart_Rate": 60,
            "Physical_Activity_Hours": 5.0
        }
        
        response = self.client.post(cluster_url, payload, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['cluster_id'], 1)

    def test_predict_cluster_invalid_input(self):
        """Ensure the serializer catches negative values and missing fields."""
        cluster_url = reverse('predict_cluster')
        payload = {
            "Age": -5, 
            # Coffee_Intake is intentionally omitted to trigger a validation error
            "Caffeine_mg": 100.0,
            "Sleep_Hours": 7.0,
            "BMI": 22.0,
            "Heart_Rate": 70,
            "Physical_Activity_Hours": 5.0
        }
        
        response = self.client.post(cluster_url, payload, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('Age', response.data)
        self.assertIn('Coffee_Intake', response.data)

class RecommendationTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.url = reverse('recommendation')
        self.valid_payload = {
            "Age": 25, "Coffee_Intake": 3.0, "Caffeine_mg": 285.0,
            "Sleep_Hours": 7.0, "BMI": 22.0, "Heart_Rate": 110, # Trigger safety!
            "Physical_Activity_Hours": 5.0, "Gender": "Female", 
            "Country": "Ukraine", "Occupation": "Student", 
            "Alcohol_Consumption": "No", "Smoking": "No"
        }

    def test_safety_guardrails(self):
        response = self.client.post(self.url, data=json.dumps(self.valid_payload), content_type='application/json')
        data = response.json()
        # With HR 110, it MUST recommend <= 0.5 cups
        self.assertLessEqual(data['recommendation']['recommended_cups'], 0.5)