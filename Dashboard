import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import requests
from datetime import datetime, timedelta
import json
import io
from typing import Dict, List
import time

# Page configuration
st.set_page_config(
    page_title="Rockfall Prediction Dashboard",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
.critical { border-left-color: #d62728 !important; }
.high { border-left-color: #ff7f0e !important; }
.medium { border-left-color: #ffbb78 !important; }
.low { border-left-color: #2ca02c !important; }
.minimal { border-left-color: #98df8a !important; }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'alert_config' not in st.session_state:
    st.session_state.alert_config = {}

def make_api_request(endpoint: str, method: str = "GET", data: Dict = None):
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Unable to connect to API. Please ensure the FastAPI server is running on localhost:8000")
        return None
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
        return None

def get_risk_color(risk_level: str) -> str:
    """Get color for risk level"""
    colors = {
        "CRITICAL": "#d62728",
        "HIGH": "#ff7f0e", 
        "MEDIUM": "#ffbb78",
        "LOW": "#2ca02c",
        "MINIMAL": "#98df8a"
    }
    return colors.get(risk_level, "#1f77b4")

def create_risk_heatmap(predictions_data: List[Dict]) -> folium.Map:
    """Create folium risk heatmap"""
    if not predictions_data:
        # Default map centered on Australia (mining country)
        m = folium.Map(location=[-25.2744, 133.7751], zoom_start=5)
        return m
    
    # Get latest prediction for each site
    site_latest = {}
    for pred in predictions_data:
        site_id = pred['site_id']
        if site_id not in site_latest or pred['timestamp'] > site_latest[site_id]['timestamp']:
            site_latest[site_id] = pred
    
    # Create map centered on mean coordinates
    lats = [float(pred.get('latitude', -25.2744)) for pred in site_latest.values()]
    lons = [float(pred.get('longitude', 133.7751)) for pred in site_latest.values()]
    
    center_lat = np.mean(lats) if lats else -25.2744
    center_lon = np.mean(lons) if lons else 133.7751
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
    
    # Add markers for each site
    for pred in site_latest.values():
        lat = float(pred.get('latitude', center_lat))
        lon = float(pred.get('longitude', center_lon))
        
        risk_level = pred.get('risk_level', 'MINIMAL')
        probability = pred.get('rockfall_probability', 0)
        
        color = get_risk_color(risk_level).replace('#', '')
        
        popup_text = f"""
        <b>Site:</b> {pred['site_id']}<br>
        <b>Risk Level:</b> {risk_level}<br>
        <b>Probability:</b> {probability:.2%}<br>
        <b>Time:</b> {pred['timestamp']}<br>
        <b>Confidence:</b> {pred.get('confidence', 0):.2%}
        """
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=max(10, probability * 20),
            popup=folium.Popup(popup_text, max_width=300),
            color=f"#{color}",
            fill=True,
            fillColor=f"#{color}",
            fillOpacity=0.7,
            weight=2
        ).add_to(m)
    
    return m

def create_time_series_chart(sensor_data: List[Dict], prediction_data: List[Dict]):
    """Create time series charts for sensor data and predictions"""
    if not sensor_data:
        return go.Figure().add_annotation(text="No sensor data available", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    df_sensors = pd.DataFrame(sensor_data)
    df_sensors['timestamp'] = pd.to_datetime(df_sensors['timestamp'])
    df_sensors = df_sensors.sort_values('timestamp')
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Temperature & Humidity', 'Rainfall & Water Content',
                       'Slope Angle & Rock Strength', 'Vibration & Wind Speed',
                       'Joint Properties', 'Rockfall Probability'),
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": True}, {"secondary_y": False}]]
    )
    
    # Temperature & Humidity
    fig.add_trace(
        go.Scatter(x=df_sensors['timestamp'], y=df_sensors['temperature'], 
                  name='Temperature (°C)', line=dict(color='red')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_sensors['timestamp'], y=df_sensors['humidity'], 
                  name='Humidity (%)', line=dict(color='blue')),
        row=1, col=1, secondary_y=True
    )
    
    # Rainfall & Water Content
    fig.add_trace(
        go.Scatter(x=df_sensors['timestamp'], y=df_sensors['rainfall'], 
                  name='Rainfall (mm)', line=dict(color='lightblue'), fill='tonexty'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df_sensors['timestamp'], y=df_sensors['water_content'], 
                  name='Water Content (%)', line=dict(color='darkblue')),
        row=1, col=2, secondary_y=True
    )
    
    # Slope & Rock Strength
    fig.add_trace(
        go.Scatter(x=df_sensors['timestamp'], y=df_sensors['slope_angle'], 
                  name='Slope Angle (°)', line=dict(color='orange')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_sensors['timestamp'], y=df_sensors['rock_strength'], 
                  name='Rock Strength (MPa)', line=dict(color='brown')),
        row=2, col=1, secondary_y=True
    )
    
    # Vibration & Wind
    fig.add_trace(
        go.Scatter(x=df_sensors['timestamp'], y=df_sensors['vibration_intensity'], 
                  name='Vibration', line=dict(color='purple')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=df_sensors['timestamp'], y=df_sensors['wind_speed'], 
                  name='Wind Speed (m/s)', line=dict(color='green')),
        row=2, col=2, secondary_y=True
    )
    
    # Joint Properties
    fig.add_trace(
        go.Scatter(x=df_sensors['timestamp'], y=df_sensors['joint_spacing'], 
                  name='Joint Spacing (cm)', line=dict(color='gray')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_sensors['timestamp'], y=df_sensors['joint_orientation'], 
                  name='Joint Orientation (°)', line=dict(color='black')),
        row=3, col=1, secondary_y=True
    )
    
    # Rockfall Probability
    if prediction_data:
        df_pred = pd.DataFrame(prediction_data)
        df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'])
        df_pred = df_pred.sort_values('timestamp')
        
        colors = [get_risk_color(risk) for risk in df_pred['risk_level']]
        
        fig.add_trace(
            go.Scatter(x=df_pred['timestamp'], y=df_pred['rockfall_probability'], 
                      name='Rockfall Probability', line=dict(color='red', width=3),
                      marker=dict(color=colors, size=8)),
            row=3, col=2
        )
        
        # Add risk level zones
        fig.add_hline(y=0.8, line_dash="dash", line_color="red", 
                     annotation_text="Critical", row=3, col=2)
        fig.add_hline(y=0.6, line_dash="dash", line_color="orange", 
                     annotation_text="High", row=3, col=2)
        fig.add_hline(y=0.4, line_dash="dash", line_color="yellow", 
                     annotation_text="Medium", row=3, col=2)
        fig.add_hline(y=0.2, line_dash="dash", line_color="lightgreen", 
                     annotation_text="Low", row=3, col=2)
    
    fig.update_layout(height=800, showlegend=True, title_text="Sensor Data and Risk Analysis")
    return fig

def create_risk_gauge(probability: float, risk_level: str):
    """Create risk gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Rockfall Risk<br><span style='font-size:0.8em;color:gray'>Level: {risk_level}</span>"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': get_risk_color(risk_level)},
            'steps': [
                {'range': [0, 20], 'color': "#98df8a"},
                {'range': [20, 40], 'color': "#ffbb78"},
                {'range': [40, 60], 'color': "#ff7f0e"},
                {'range': [60, 80], 'color': "#d62728"},
                {'range': [80, 100], 'color': "#8b0000"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_contributing_factors_chart(factors: Dict[str, float]):
    """Create contributing factors bar chart"""
    if not factors:
        return go.Figure().add_annotation(text="No factors data available", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
    factor_names, factor_values = zip(*sorted_factors)
    
    fig = go.Figure(data=[
        go.Bar(x=factor_values, y=factor_names, orientation='h',
               marker_color='lightblue', text=[f'{v:.1%}' for v in factor_values],
               textposition='auto')
    ])
    
    fig.update_layout(
        title="Contributing Factors to Risk",
        xaxis_title="Importance",
        height=400,
        margin=dict(l=150, r=20, t=40, b=20)
    )
    
    return fig

# Main Dashboard
def main():
    st.title("⛰️ Rockfall Prediction Dashboard")
    st.markdown("AI-powered monitoring system for open-pit mines")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox("Select Page", 
                           ["Overview", "Site Analysis", "Data Upload", "Alert Configuration", "System Health"])
        
        st.header("Refresh Controls")
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
        if st.button("Manual Refresh") or auto_refresh:
            st.session_state.last_update = datetime.now()
            if auto_refresh:
                time.sleep(1)
                st.rerun()
    
    # Get sites data
    sites_data = make_api_request("/sites")
    available_sites = []
    if sites_data:
        available_sites = [site['site_id'] for site in sites_data['sites']]
    
    if page == "Overview":
        st.header("System Overview")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_sites = len(available_sites) if available_sites else 0
            st.metric("Active Sites", total_sites)
        
        with col2:
            # Get latest predictions for all sites
            all_predictions = []
            for site in available_sites[:5]:  # Limit for demo
                pred_data = make_api_request(f"/predictions/{site}?limit=1")
                if pred_data and pred_data['predictions']:
                    all_predictions.extend(pred_data['predictions'])
            
            high_risk_count = sum(1 for p in all_predictions if p.get('risk_level') in ['HIGH', 'CRITICAL'])
            st.metric("High Risk Sites", high_risk_count)
        
        with col3:
            avg_prob = np.mean([p.get('rockfall_probability', 0) for p in all_predictions]) if all_predictions else 0
            st.metric("Average Risk", f"{avg_prob:.1%}")
        
        with col4:
            st.metric("Last Update", 
                     st.session_state.last_update.strftime("%H:%M:%S") if st.session_state.last_update else "Never")
        
        # Risk heatmap
        st.subheader("Risk Heatmap")
        if all_predictions:
            risk_map = create_risk_heatmap(all_predictions)
            st_folium(risk_map, width=700, height=400)
        else:
            st.info("No prediction data available for mapping")
        
        # Recent alerts
        st.subheader("Recent High-Risk Alerts")
        high_risk_alerts = [p for p in all_predictions if p.get('risk_level') in ['HIGH', 'CRITICAL']]
        
        if high_risk_alerts:
            alert_df = pd.DataFrame(high_risk_alerts)
            alert_df['probability_pct'] = alert_df['rockfall_probability'].apply(lambda x: f"{x:.1%}")
            st.dataframe(alert_df[['site_id', 'timestamp', 'risk_level', 'probability_pct', 'confidence']],
                        use_container_width=True)
        else:
            st.success("No high-risk alerts currently active")
    
    elif page == "Site Analysis":
        st.header("Detailed Site Analysis")
        
        if not available_sites:
            st.warning("No sites available. Please upload data first.")
            return
        
        selected_site = st.selectbox("Select Site", available_sites)
        
        if selected_site:
            # Get site data
            predictions_data = make_api_request(f"/predictions/{selected_site}?limit=50")
            sensor_data = make_api_request(f"/sensors/{selected_site}?limit=100")
            
            if predictions_data and predictions_data['predictions']:
                latest_prediction = predictions_data['predictions'][0]
                
                # Current risk status
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Current Risk Status")
                    risk_gauge = create_risk_gauge(
                        latest_prediction['rockfall_probability'],
                        latest_prediction['risk_level']
                    )
                    st.plotly_chart(risk_gauge, use_container_width=True)
                    
                    # Key metrics
                    st.markdown(f"""
                    <div class="metric-card {latest_prediction['risk_level'].lower()}">
                        <h4>Site: {selected_site}</h4>
                        <p><strong>Risk Level:</strong> {latest_prediction['risk_level']}</p>
                        <p><strong>Probability:</strong> {latest_prediction['rockfall_probability']:.1%}</p>
                        <p><strong>Confidence:</strong> {latest_prediction['confidence']:.1%}</p>
                        <p><strong>Last Update:</strong> {latest_prediction['timestamp']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.subheader("Contributing Factors")
                    factors_chart = create_contributing_factors_chart(latest_prediction['contributing_factors'])
                    st.plotly_chart(factors_chart, use_container_width=True)
                
                # Recommendations
                st.subheader("Recommended Actions")
                for i, rec in enumerate(latest_prediction['recommendations'], 1):
                    st.write(f"{i}. {rec}")
                
                # Historical trends
                st.subheader("Historical Data and Trends")
                if sensor_data and sensor_data['data']:
                    time_series_chart = create_time_series_chart(
                        sensor_data['data'], predictions_data['predictions']
                    )
                    st.plotly_chart(time_series_chart, use_container_width=True)
                else:
                    st.info("No historical sensor data available")
                
                # Prediction history table
                st.subheader("Prediction History")
                pred_df = pd.DataFrame(predictions_data['predictions'])
                pred_df['probability_pct'] = pred_df['rockfall_probability'].apply(lambda x: f"{x:.1%}")
                st.dataframe(pred_df[['timestamp', 'risk_level', 'probability_pct', 'confidence']],
                           use_container_width=True)
            else:
                st.info("No predictions available for this site")
    
    elif page == "Data Upload":
        st.header("Data Upload and Management")
        
        # File upload section
        st.subheader("Upload Sensor Data")
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file:
            # Preview data
            df = pd.read_csv(uploaded_file)
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("Upload Data"):
                try:
                    files = {"file": uploaded_file.getvalue()}
                    response = requests.post(f"{API_BASE_URL}/upload/sensors", 
                                           files={"file": ("data.csv", uploaded_file.getvalue(), "text/csv")})
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"Successfully uploaded {result['count']} records!")
                    else:
                        st.error(f"Upload failed: {response.text}")
                        
                except Exception as e:
                    st.error(f"Upload error: {str(e)}")
        
        # Manual data entry
        st.subheader("Manual Data Entry")
        with st.form("manual_entry"):
            col1, col2 = st.columns(2)
            
            with col1:
                site_id = st.text_input("Site ID", value="SITE_001")
                slope_angle = st.number_input("Slope Angle (°)", 0.0, 90.0, 45.0)
                rock_strength = st.number_input("Rock Strength (MPa)", 0.0, 100.0, 50.0)
                joint_spacing = st.number_input("Joint Spacing (cm)", 0.0, 200.0, 30.0)
                joint_orientation = st.number_input("Joint Orientation (°)", 0.0, 360.0, 90.0)
                water_content = st.number_input("Water Content (%)", 0.0, 100.0, 20.0)
                latitude = st.number_input("Latitude", -90.0, 90.0, -25.2744)
            
            with col2:
                temperature = st.number_input("Temperature (°C)", -50.0, 60.0, 25.0)
                humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
                wind_speed = st.number_input("Wind Speed (m/s)", 0.0, 50.0, 5.0)
                rainfall = st.number_input("Rainfall (mm)", 0.0, 200.0, 0.0)
                vibration_intensity = st.number_input("Vibration Intensity", 0.0, 10.0, 2.0)
                longitude = st.number_input("Longitude", -180.0, 180.0, 133.7751)
            
            if st.form_submit_button("Submit Data & Get Prediction"):
                sensor_data = {
                    "site_id": site_id,
                    "slope_angle": slope_angle,
                    "rock_strength": rock_strength,
                    "joint_spacing": joint_spacing,
                    "joint_orientation": joint_orientation,
                    "water_content": water_content,
                    "temperature": temperature,
                    "humidity": humidity,
                    "wind_speed": wind_speed,
                    "rainfall": rainfall,
                    "vibration_intensity": vibration_intensity,
                    "latitude": latitude,
                    "longitude": longitude
                }
                
                result = make_api_request("/predict", method="POST", data=sensor_data)
                if result:
                    st.success("Data submitted and prediction generated!")
                    
                    # Display prediction
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Risk Level", result['risk_level'])
                        st.metric("Probability", f"{result['rockfall_probability']:.1%}")
                    
                    with col2:
                        st.metric("Confidence", f"{result['confidence']:.1%}")
                        st.write("**Recommendations:**")
                        for rec in result['recommendations']:
                            st.write(f"• {rec}")
    
    elif page == "Alert Configuration":
        st.header("Alert Configuration")
        
        # Get current config
        current_config = make_api_request("/alerts/config")
        if current_config:
            st.session_state.alert_config = current_config
        
        with st.form("alert_config"):
            st.subheader("Email Configuration")
            email_enabled = st.checkbox("Enable Email Alerts", 
                                       value=st.session_state.alert_config.get('email_enabled', True))
            
            risk_threshold = st.slider("Risk Threshold for Alerts", 0.0, 1.0, 
                                     st.session_state.alert_config.get('risk_threshold', 0.7), 0.1)
            
            email_recipients = st.text_area("Email Recipients (one per line)", 
                                          value="\n".join(st.session_state.alert_config.get('email_recipients', [])))
            
            col1, col2 = st.columns(2)
            with col1:
                smtp_server = st.text_input("SMTP Server", 
                                          value=st.session_state.alert_config.get('smtp_server', 'smtp.gmail.com'))
                smtp_port = st.number_input("SMTP Port", 1, 65535, 
                                          st.session_state.alert_config.get('smtp_port', 587))
            
            with col2:
                smtp_username = st.text_input("SMTP Username", 
                                            value=st.session_state.alert_config.get('smtp_username', ''))
                smtp_password = st.text_input("SMTP Password", type="password")
            
            if st.form_submit_button("Update Configuration"):
                config_data = {
                    "email_enabled": email_enabled,
                    "sms_enabled": False,
                    "risk_threshold": risk_threshold,
                    "email_recipients": [email.strip() for email in email_recipients.split('\n') if email.strip()],
                    "smtp_server": smtp_server,
                    "smtp_port": smtp_port,
                    "smtp_username": smtp_username,
                    "smtp_password": smtp_password if smtp_password else st.session_state.alert_config.get('smtp_password', '')
                }
                
                result = make_api_request("/alerts/config", method="POST", data=config_data)
                if result:
                    st.success("Alert configuration updated!")
                    st.session_state.alert_config = config_data
    
    elif page == "System Health":
        st.header("System Health Monitor")
        
        health_data = make_api_request("/health")
        if health_data:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status_color = "🟢" if health_data['status'] == 'healthy' else "🔴"
                st.metric("System Status", f"{status_color} {health_data['status'].title()}")
            
            with col2:
                model_status = "🟢 Loaded" if health_data['model_loaded'] else "🔴 Not Loaded"
                st.metric("ML Model", model_status)
            
            with col3:
                db_status = "🟢 Connected" if health_data['database_connected'] else "🔴 Disconnected"
                st.metric("Database", db_status)
            
            st.subheader("System Information")
            st.json(health_data)
        
        # API endpoints test
        st.subheader("API Endpoints Test")
        endpoints = ["/", "/sites", "/health"]
        
        for endpoint in endpoints:
            with st.expander(f"Test {endpoint}"):
                if st.button(f"Test {endpoint}", key=f"test_{endpoint.replace('/', '_')}"):
                    result = make_api_request(endpoint)
                    if result:
                        st.success(f"✅ {endpoint} - Response received")
                        st.json(result)
                    else:
                        st.error(f"❌ {endpoint} - Failed")
        
        # Generate sample data button
        st.subheader("Sample Data Generation")
        if st.button("Generate Sample Data"):
            sample_sites = ["SITE_001", "SITE_002", "SITE_003"]
            
            for site in sample_sites:
                # Generate random sensor data
                sensor_data = {
                    "site_id": site,
                    "slope_angle": np.random.uniform(30, 70),
                    "rock_strength": np.random.uniform(20, 80),
                    "joint_spacing": np.random.uniform(10, 50),
                    "joint_orientation": np.random.uniform(0, 360),
                    "water_content": np.random.uniform(10, 40),
                    "temperature": np.random.uniform(15, 35),
                    "humidity": np.random.uniform(40, 80),
                    "wind_speed": np.random.uniform(0, 15),
                    "rainfall": np.random.uniform(0, 10),
                    "vibration_intensity": np.random.uniform(0, 5),
                    "latitude": -25.2744 + np.random.uniform(-0.1, 0.1),
                    "longitude": 133.7751 + np.random.uniform(-0.1, 0.1)
                }
                
                result = make_api_request("/predict", method="POST", data=sensor_data)
                if result:
                    st.success(f"Generated sample data for {site}")
                else:
                    st.error(f"Failed to generate data for {site}")

if __name__ == "__main__":
    main()
