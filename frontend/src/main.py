import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os

# --- Configurations ---
st.set_page_config(page_title="Coffee Persona Analyzer", page_icon="☕", layout="wide")
API_URL = "http://127.0.0.1:8000/api/cluster/"

# --- Load Background Data for the Plot ---
# We use st.cache_data so it only loads the CSV once to keep the app lightning fast!
@st.cache_data
def load_cluster_data():
    # Adjust path assuming you run this from the 'frontend' folder
    file_path = os.path.join(os.path.dirname(__file__), '../../data/processed/features_engineered_with_clusters.csv')
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

df = load_cluster_data()

# --- User Interface ---
st.title("☕ Coffee Persona Analyzer")
st.markdown("Enter your daily habits below to discover your true Coffee Persona.")

# Create a clean layout with columns
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Your Stats")
    age = st.number_input("Age", min_value=18, max_value=100, value=25)
    coffee_cups = st.slider("Coffee Cups per Day", 0.0, 10.0, 2.0, 0.5)
    caffeine_mg = coffee_cups * 95  # Estimate 95mg per standard cup
    
    sleep = st.slider("Sleep Hours", 2.0, 12.0, 7.0, 0.5)
    bmi = st.number_input("BMI", min_value=15.0, max_value=40.0, value=22.0)
    heart_rate = st.slider("Resting Heart Rate", 40, 120, 70)
    activity = st.slider("Physical Activity (Hours/Week)", 0.0, 20.0, 5.0, 0.5)

    analyze_button = st.button("🔮 Discover My Persona", use_container_width=True)

with col2:
    if analyze_button:
        # 1. Package the data for Django
        payload = {
            "Age": age,
            "Coffee_Intake": coffee_cups,
            "Caffeine_mg": caffeine_mg,
            "Sleep_Hours": sleep,
            "BMI": bmi,
            "Heart_Rate": heart_rate,
            "Physical_Activity_Hours": activity
        }
        
        # 2. Call your new API!
        with st.spinner("Consulting the Sorting Hat..."):
            try:
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    persona_name = result['profile']['name']
                    persona_desc = result['profile']['description']
                    
                    st.success(f"### You are: **{persona_name}**")
                    st.write(f"*{persona_desc}*")
                    
                    # 3. Show the interactive scatter plot
                    if df is not None:
                        st.subheader("Where you fit in:")
                        # Convert Cluster column to string so Plotly uses distinct colors instead of a gradient
                        persona_map = {
                            0: "The High-Octane Chugger",
                            1: "The Decaf Abstainer",
                            2: "The Balanced Brewer"
                        }
                        df['Cluster_Label'] = df['Cluster'].map(persona_map)
                        
                        fig = px.scatter(
                            df, x='PCA1', y='PCA2', color='Cluster_Label',
                            title="The Global Coffee Personas",
                            labels={'PCA1': 'Metabolic & Sleep Axis', 'PCA2': 'Caffeine Volume Axis'},
                            color_discrete_sequence=px.colors.qualitative.Vivid
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Cluster data file not found.")
                        
                else:
                    st.error(f"API Error: {response.status_code} - Make sure your Django server is running!")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend.")