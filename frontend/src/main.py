import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os
import plotly.graph_objects as go

"""
Main Streamlit interface for the Coffee Persona Analyzer.
This module orchestrates data visualization, API communication, and user interaction.
"""

# Page configuration
st.set_page_config(page_title="Coffee Persona Analyzer", layout="wide")

# API Endpoints
API_BASE_URL = 'http://127.0.0.1:8000/api'
CLUSTER_API_URL = f'{API_BASE_URL}/cluster/'
PREDICT_API_URL = f'{API_BASE_URL}/predict/state/'
RECOMMEND_API_URL = f'{API_BASE_URL}/recommendation/'
CAFFEINE_PER_STANDARD_CUP = 95.0

@st.cache_data
def load_cluster_data():
    """Loads preprocessed dataset for visualization and comparative analysis."""
    file_path = os.path.join(os.path.dirname(__file__), '../../data/processed/features_engineered_with_clusters.csv')
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

df = load_cluster_data()

def fetch_explanation(prediction_id):
    """Retrieves SHAP-based feature importance from the backend for a specific record."""
    url = f"{API_BASE_URL}/explain/{prediction_id}/"
    try:
        response = requests.get(url)
        return response.json() if response.status_code == 200 else None
    except requests.RequestException:
        return None

def display_explanation_panel(prediction_id):
    """
    Renders an interactive panel displaying SHAP-based feature importance 
    using tabs for distinct health models.
    """
    st.markdown("---")
    st.subheader("Model Interpretation")
    st.write("Understand the key factors influencing your health metrics via SHAP analysis.")

    with st.spinner("Retrieving model insights..."):
        data = fetch_explanation(prediction_id)
    
    if not data or "explanations" not in data:
        st.warning("Insight data is currently unavailable.")
        return

    explanations = data["explanations"]
    tab_sleep, tab_stress, tab_health = st.tabs(["Sleep Quality", "Stress Level", "Health Issues"])
    
    # Mapping for streamlined iteration
    target_configs = [
        ("sleep_quality", tab_sleep), 
        ("stress_level", tab_stress), 
        ("health_issues", tab_health)
    ]

    for key, tab in target_configs:
        with tab:
            model_data = explanations.get(key, {})
            drivers = model_data.get("top_drivers", [])
            
            if not drivers:
                st.info("No significant predictive drivers identified.")
                continue

            col_a, col_b = st.columns([1, 1])
            
            with col_a:
                st.markdown("##### Key Drivers")
                for driver in drivers:
                    feature = driver['feature'].title()
                    # Assign color-coded feedback based on feature impact
                    if driver["impact"] == "positive":
                        st.success(f"**{feature}**: {driver['display_text']}")
                    else:
                        st.warning(f"**{feature}**: {driver['display_text']}")
            
            with col_b:
                st.markdown("##### Impact Breakdown")
                df_viz = pd.DataFrame(drivers)
                df_viz = df_viz.sort_values(by="shap_influence", ascending=True)
                
                # Assign colors: Red for positive impact, Blue for negative
                colors = ['#ef553b' if x > 0 else '#636efa' for x in df_viz["shap_influence"]]
                
                fig = px.bar(
                    df_viz, 
                    x="shap_influence", 
                    y="feature", 
                    orientation="h",
                    labels={"shap_influence": "Influence Score", "feature": ""}
                )
                
                fig.update_traces(marker_color=colors)
                fig.add_vline(x=0, line_width=1, line_color="white")
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

st.title("Coffee Persona Analyzer")
st.markdown("Enter your daily habits to discover your Coffee Persona and receive AI-driven health insights.")

col_input, col_results = st.columns([1, 2])

with col_input:
    st.markdown("#### Biometrics")
    age = st.number_input("Age", min_value=18, max_value=100, value=25)
    bmi = st.number_input("BMI", min_value=15.0, max_value=40.0, value=22.0)
    heart_rate = st.slider("Resting Heart Rate (bpm)", 40, 120, 70)
    
    st.markdown("#### Lifestyle")
    coffee_cups = st.slider("Coffee Cups per Day", 0.0, 10.0, 2.0, 0.5)
    caffeine_mg = coffee_cups * CAFFEINE_PER_STANDARD_CUP
    sleep = st.slider("Sleep Hours", 2.0, 12.0, 7.0, 0.5)
    activity = st.slider("Physical Activity (Hours/Week)", 0.0, 20.0, 5.0, 0.5)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])

    st.markdown("#### Demographics")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    occupation = st.selectbox("Occupation", ["Other", "Service", "Office", "Student", "Healthcare"])
    country = st.selectbox("Country", [
        "Germany", "Brazil", "Spain", "Mexico", "France", "Canada", "UK", "Switzerland",
        "Netherlands", "Italy", "China", "Japan", "Belgium", "Finland", "Australia",
        "USA", "Sweden", "India", "Norway", "South Korea"
    ])

    analyze_button = st.button("Discover My Persona", use_container_width=True, type="primary")

with col_results:
    if analyze_button:
        payload = {
            "Age": age, "Coffee_Intake": coffee_cups, "Caffeine_mg": caffeine_mg,
            "Sleep_Hours": sleep, "BMI": bmi, "Heart_Rate": heart_rate,
            "Physical_Activity_Hours": activity, "Gender": gender, "Country": country,
            "Occupation": occupation, "Alcohol_Consumption": alcohol, "Smoking": smoking
        }

        with st.spinner("Analyzing physiological patterns..."):
            anomaly_res = requests.post(f"{API_BASE_URL}/anomalies/", json=payload)
            
            if anomaly_res.status_code == 200 and anomaly_res.json().get('is_anomaly'):
                st.error("Physiological Pattern Alert")
                st.warning("""
                    Your biometric profile (Heart Rate, BMI, and Intake) is statistically unusual. 
                    The following insights should be treated with caution. If you have concerns 
                    about your resting heart rate or current health status, please consult a 
                    healthcare professional.
                """)
        
        with st.spinner("Consulting inference engines..."):
            try:
                # 1. Clustering Analysis
                response = requests.post(CLUSTER_API_URL, json=payload, timeout=5)
                response.raise_for_status() 
                result = response.json()
                
                st.success(f"### Persona Identified: **{result['profile']['name']}**")
                st.write(f"*{result['profile']['description']}*")
                
                # 2. Store prediction context for SHAP explanation
                predict_response = requests.post(PREDICT_API_URL, json=payload, timeout=5)
                if predict_response.status_code == 200:
                    st.session_state['prediction_id'] = predict_response.json().get("prediction_id")
                
                # 3. Visualize global persona clusters
                if df is not None:
                    color_map = {'0': '#e76f51', '1': '#2a9d8f', '2': '#e9c46a'}
                    label_map = {'0': 'High Risk', '1': 'Optimal', '2': 'Moderate'}
                    
                    fig = px.scatter(df, x='UMAP1', y='UMAP2', title="Global Coffee Habit Clusters")
                    
                    # Apply categorical coloring
                    fig.update_traces(marker=dict(color=df['Cluster'].astype(str).map(color_map), size=8, opacity=0.8))

                    # Append legend entries
                    for cluster_id, color in color_map.items():
                        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', 
                                                 marker=dict(color=color, size=8), name=label_map[cluster_id]))

                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(showgrid=True, gridcolor='lightgrey', showticklabels=False),
                        yaxis=dict(showgrid=True, gridcolor='lightgrey', showticklabels=False)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Cluster visualization data not found.")
                
                st.markdown("---")
                st.subheader("Personalized Health Recommendation")
                
                with st.spinner("Optimizing caffeine intake..."):
                    rec_response = requests.post(RECOMMEND_API_URL, json=payload, timeout=5)
                    if rec_response.status_code == 200:
                        rec = rec_response.json().get('recommendation', {})
    
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Recommended", f"{rec.get('recommended_cups')} cups", f"{rec.get('delta')} cups")
                        c2.metric("Stress Risk Reduction", f"-{rec.get('stress_reduction_pct', 0)}%")
                        c3.metric("Sleep Impact", rec.get('impact_sleep', 'Stable'))

                        st.info(f"**AI Health Insight:** {rec.get('reasoning', 'Recommendation generated.')}")
                    else:
                        st.warning("Recommendation engine is temporarily unreachable.")
                    
            except Exception as e:
                st.error(f"Analysis process interrupted: {e}")

# Render SHAP interpretability panel
if st.session_state.get('prediction_id'):
    display_explanation_panel(st.session_state['prediction_id'])