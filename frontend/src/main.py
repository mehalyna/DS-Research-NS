import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os

# --- Configurations ---
st.set_page_config(page_title="Coffee Persona Analyzer", layout="wide")

CLUSTER_API_URL = 'http://127.0.0.1:8000/api/cluster/'
PREDICT_API_URL = 'http://127.0.0.1:8000/api/predict/state/'
RECOMMEND_API_URL = 'http://127.0.0.1:8000/api/recommendation/'
CAFFEINE_PER_STANDARD_CUP = 95.0

@st.cache_data
def load_cluster_data():
    file_path = os.path.join(os.path.dirname(__file__), '../../data/processed/features_engineered_with_clusters.csv')
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

df = load_cluster_data()

# --- Explanation Helper Functions ---
def fetch_explanation(prediction_id):
    url = f"http://127.0.0.1:8000/api/explain/{prediction_id}/" 
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None

def display_explanation_panel(prediction_id):
    st.markdown("---")
    st.subheader("Why did you get this prediction?")
    st.write("Understand the key factors driving your health scores based on SHAP AI Analysis.")

    with st.spinner("Generating SHAP explanations..."):
        data = fetch_explanation(prediction_id)
    
    if not data or "explanations" not in data:
        st.warning("Could not retrieve explanation data.")
        return

    explanations = data["explanations"]
    tab1, tab2, tab3 = st.tabs(["Sleep Quality", "Stress Level", "Health Issues"])
    
    targets = [("sleep_quality", tab1), ("stress_level", tab2), ("health_issues", tab3)]
    
    def clean_text(text):
        return text.replace("num__", "").replace("cat__", "").replace("_", " ")

    for key, tab in targets:
        with tab:
            model_data = explanations.get(key, {})
            drivers = model_data.get("top_drivers", [])
            
            if not drivers:
                st.info("No significant drivers found.")
                continue

            colA, colB = st.columns([1, 1])
            
            with colA:
                st.markdown("##### Key Drivers")
                for driver in drivers:
                    clean_name = clean_text(driver['feature']).title()
                    clean_sentence = clean_text(driver['display_text'])
                    
                    if driver["impact"] == "positive":
                        st.success(f"**{clean_name}**: {clean_sentence}")
                    else:
                        st.warning(f"**{clean_name}**: {clean_sentence}")
            
            with colB:
                st.markdown("##### Visual Impact Breakdown")
                chart_data = pd.DataFrame(drivers)
                
                # 1. Clean labels and convert numbers
                chart_data["Feature"] = chart_data["feature"].apply(lambda x: clean_text(x).title())
                chart_data["shap_influence"] = pd.to_numeric(chart_data["shap_influence"])
                
                # 2. Sort so the biggest impact is at the top
                chart_data = chart_data.sort_values(by="shap_influence", ascending=True)
                
                # 3. Create a simple color list based on the values
                # Red for positive, Blue for negative
                colors = ['#ef553b' if x > 0 else '#636efa' for x in chart_data["shap_influence"]]
                
                # 4. Use a standard bar chart and pass the color list directly
                fig = px.bar(
                    chart_data, 
                    x="shap_influence", 
                    y="Feature", 
                    orientation="h"
                )
                
                # Update the traces to use our custom color list
                fig.update_traces(marker_color=colors)
                
                # 5. Add a vertical zero-line and clean layout
                fig.add_vline(x=0, line_width=2, line_color="white", opacity=0.5)
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', 
                    xaxis_title="Impact on Score", 
                    yaxis_title="",
                    showlegend=False
                )
                st.plotly_chart(fig, width='stretch')

# --- Main UI ---
st.title("Coffee Persona Analyzer")
st.markdown("Enter your daily habits below to discover your true Coffee Persona and get AI-powered health insights.")

col1, col2 = st.columns([1, 2])

with col1:
    # Section 1: Biometrics
    st.markdown("#### Core Biometrics")
    age = st.number_input("Age", min_value=18, max_value=100, value=25)
    bmi = st.number_input("BMI", min_value=15.0, max_value=40.0, value=22.0)
    heart_rate = st.slider("Resting Heart Rate", 40, 120, 70)
    
    # Section 2: Lifestyle & Habits
    st.markdown("#### Lifestyle & Habits")
    coffee_cups = st.slider("Coffee Cups per Day", 0.0, 10.0, 2.0, 0.5)
    caffeine_mg = coffee_cups * CAFFEINE_PER_STANDARD_CUP
    sleep = st.slider("Sleep Hours", 2.0, 12.0, 7.0, 0.5)
    activity = st.slider("Physical Activity (Hours/Week)", 0.0, 20.0, 5.0, 0.5)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])

    # Section 3: Demographics
    st.markdown("#### Demographics")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    occupation = st.selectbox("Occupation", ["Other", "Service", "Office", "Student", "Healthcare"])
    country = st.selectbox("Country", [
        "Germany", "Brazil", "Spain", "Mexico", "France", "Canada", "UK", "Switzerland",
        "Netherlands", "Italy", "China", "Japan", "Belgium", "Finland", "Australia",
        "USA", "Sweden", "India", "Norway", "South Korea"
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_button = st.button("Discover My Persona", width='stretch', type="primary")

with col2:
    if analyze_button:
        payload = {
            "Age": age, "Coffee_Intake": coffee_cups, "Caffeine_mg": caffeine_mg,
            "Sleep_Hours": sleep, "BMI": bmi, "Heart_Rate": heart_rate,
            "Physical_Activity_Hours": activity, "Gender": gender, "Country": country,
            "Occupation": occupation, "Alcohol_Consumption": alcohol, "Smoking": smoking
        }

        with st.spinner("Checking safety bounds..."):
            anomaly_res = requests.post("http://127.0.0.1:8000/api/anomalies/", json=payload)
            if anomaly_res.status_code == 200 and anomaly_res.json().get('is_anomaly'):
                st.error("Physiological Outlier Detected")
                st.warning("""
                    Your combined biometrics (Heart Rate, BMI, and Intake) fall outside of typical training ranges. 
                    The following health predictions and recommendations should be treated with extreme caution.
                    Your data pattern is highly unusual.
                    We recommend manually checking your resting heart rate and consulting a healthcare provider before increasing caffeine intake.
                """)
        
        with st.spinner("Consulting the ML models..."):
            try:
                # 1. Cluster Call
                response = requests.post(CLUSTER_API_URL, json=payload, timeout=5)
                response.raise_for_status() 
                result = response.json()
                
                st.success(f"### You are: **{result['profile']['name']}**")
                st.write(f"*{result['profile']['description']}*")
                
                # 2. Predict Call
                predict_response = requests.post(PREDICT_API_URL, json=payload, timeout=5)
                if predict_response.status_code == 200:
                    st.session_state['prediction_id'] = predict_response.json().get("prediction_id")
                
                # 3. Restored, beautiful Plotly Chart
                if df is not None:
                    st.subheader("Where you fit in:")
                    
                    # Create a simple scatter plot with NO color mapping to avoid errors
                    fig = px.scatter(
                        df, 
                        x='UMAP1', 
                        y='UMAP2', 
                        title="The Global Coffee Personas"
                    )
                    
                    # Force all points to be the same professional blue color
                    fig.update_traces(marker=dict(color='#636efa', size=5, opacity=0.6))
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis_title="Similarity Dimension 1",
                        yaxis_title="Similarity Dimension 2"
                    )
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.warning("Cluster data file not found.")
                
                st.markdown("---")
                st.subheader("Personal Health Recommendation")
                
                with st.spinner("Calculating optimal intake..."):
                    rec_response = requests.post(RECOMMEND_API_URL, json=payload, timeout=5)
                    if rec_response.status_code == 200:
                        rec_data = rec_response.json()
                        rec = rec_data['recommendation']
    
                        # Use columns to make it look professional
                        col_met1, col_met2, col_met3 = st.columns(3)
                        
                        with col_met1:
                            st.metric("Recommended", f"{rec['recommended_cups']} cups", f"{rec['delta']} cups")
                        
                        with col_met2:
                            # Show the Stress Reduction % we calculated
                            reduction = rec.get('stress_reduction_pct', 0)
                            st.metric("Stress Risk Reduction", f"-{reduction}%", delta_color="normal")
                            
                        with col_met3:
                            # Show the Sleep Impact
                            sleep_status = rec.get('impact_sleep', 'Stable')
                            st.metric("Sleep Quality", sleep_status)

                        # Display the dynamic "Human-Speak" reasoning we just perfected
                        st.info(f"**AI Health Insight:** {rec.get('reasoning', 'No specific insight available.')}")
                        
                    else:
                        st.warning("Recommendation engine currently unavailable.")
                    
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# --- Render the Explanation Panel Full-Width at the Bottom ---
if 'prediction_id' in st.session_state and st.session_state['prediction_id']:
    display_explanation_panel(st.session_state['prediction_id'])