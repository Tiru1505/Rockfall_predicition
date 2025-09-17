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
import asyncio

# Page configuration
st.set_page_config(
    page_title="Rockfall Prediction Dashboard",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
    margin: 0.5rem 0;
}
.critical { 
    border-left-color: #d62728 !important; 
    background-color: #ffe6e6 !important;
    animation: pulse 2s infinite;
}
.high { 
    border-left-color: #ff7f0e !important; 
    background-color: #fff4e6 !important;
}
.medium { 
    border-left-color: #ffbb78 !important; 
    background-color: #fffbf0 !important;
}
.low { 
    border-left-color: #2ca02c !important; 
    background-color: #f0fff0 !important;
}
.minimal { 
    border-left-color: #98df8a !important; 
    background-color: #f8fff8 !important;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.7; }
    100% { opacity: 1; }
}

.alert-banner {
    background-color: #ff4444;
    color: white;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
    text-align: center;
    font-weight: bold;
    animation: blink 1s infinite;
}

@keyframes blink {
    0% { background-color: #ff4444; }
    50% { background-color: #ff6666; }
    100% { background-color: #ff4444; }
}

.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
}

.status-active { background-color: #00ff00; }
.status-warning { background-color: #ffaa00; }
.status-critical { background-color: #ff0000; }

.live-data {
    border: 2px solid #00ff00;
    border-radius: 8px;
    padding: 10px;
    margin: 5px 0;
    background: linear-gradient(90deg, rgba(0,255,0,0.1) 0%, rgba(0,255,0,0.05) 100%);
}
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'alert_config' not in st.session_state:
    st.session_state.alert_config = {}
if 'live_alerts' not in st.session_state:
    st.session_state.live_alerts = []
if 'dashboard_stats' not in st.session_state:
    st.session_state.dashboard_stats = {}

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
        st.error("⚠️ Unable to connect to API. Please ensure the FastAPI server is running on localhost:8000")
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

def create_live_alert_banner():
    """Create live alert banner for critical situations"""
    alerts_data = make_api_request("/alerts/recent?limit=5")
    if alerts_data and alerts_data['alerts']:
        critical_alerts = [
            alert for alert in alerts_data['alerts']
            if alert['risk_level'] in ['CRITICAL', 'HIGH']
            and (datetime.now() - datetime.fromisoformat(alert['timestamp'].replace('Z', '+00:00'))).total_seconds() < 3600
        ]

        if critical_alerts:
            alert_count = len(critical_alerts)
            st.markdown(f"""
            <div class="alert-banner">
                🚨 ACTIVE ALERTS: {alert_count} high-risk situation(s) detected! 
                Latest: {critical_alerts[0]['site_id']} at {critical_alerts[0]['timestamp'][:16]}
            </div>
            """, unsafe_allow_html=True)

            # Show detailed alerts in expander
            with st.expander(f"⚠️ View {alert_count} Active Alert(s)", expanded=False):
                for alert in critical_alerts:
                    risk_class = alert['risk_level'].lower()
                    st.markdown(f"""
                    <div class="metric-card {risk_class}">
                        <strong>{alert['site_id']}</strong> - {alert['risk_level']} RISK<br>
                        <small>Probability: {alert['probability']:.1%} | Time: {alert['timestamp'][:16]}</small><br>
                        <small>{alert['message']}</small>
                    </div>
                    """, unsafe_allow_html=True)

def create_live_stats_cards():
    """Create live statistics cards"""
    stats = make_api_request("/dashboard/stats")
    if not stats:
        return

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        status_color = "status-active" if stats['total_sites'] > 0 else "status-warning"
        st.markdown(f"""
        <div class="live-data">
            <span class="status-indicator {status_color}"></span>
            <strong>Active Sites</strong><br>
            <span style="font-size: 24px; font-weight: bold;">{stats['total_sites']}</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        status_color = "status-critical" if stats['high_risk_sites'] > 0 else "status-active"
        st.markdown(f"""
        <div class="live-data">
            <span class="status-indicator {status_color}"></span>
            <strong>High Risk Sites</strong><br>
            <span style="font-size: 24px; font-weight: bold; color: {'#ff4444' if stats['high_risk_sites'] > 0 else '#00aa00'}">{stats['high_risk_sites']}</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        status_color = "status-critical" if stats['recent_alerts'] > 0 else "status-active"
        st.markdown(f"""
        <div class="live-data">
            <span class="status-indicator {status_color}"></span>
            <strong>Recent Alerts</strong><br>
            <span style="font-size: 24px; font-weight: bold; color: {'#ff4444' if stats['recent_alerts'] > 0 else '#00aa00'}">{stats['recent_alerts']}</span>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        avg_risk = stats['average_risk']
        risk_color = "#ff4444" if avg_risk > 0.6 else "#ffaa00" if avg_risk > 0.3 else "#00aa00"
        st.markdown(f"""
        <div class="live-data">
            <span class="status-indicator status-active"></span>
            <strong>Average Risk</strong><br>
            <span style="font-size: 24px; font-weight: bold; color: {risk_color}">{avg_risk:.1%}</span>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        last_update = datetime.fromisoformat(stats['last_update'].replace('Z', '+00:00'))
        time_diff = (datetime.now() - last_update).total_seconds()
        status_color = "status-active" if time_diff < 60 else "status-warning"
        st.markdown(f"""
        <div class="live-data">
            <span class="status-indicator {status_color}"></span>
            <strong>Last Update</strong><br>
            <span style="font-size: 16px; font-weight: bold;">{last_update.strftime('%H:%M:%S')}</span><br>
            <small>{int(time_diff)}s ago</small>
        </div>
        """, unsafe_allow_html=True)

def create_risk_heatmap_live():
    """Create live risk heatmap with real-time data"""
    site_data = make_api_request("/live/sites")
    if not site_data or not site_data['sites']:
        st.warning("No live site data available")
        return None

    sites = site_data['sites']

    # Calculate map center
    lats = [site['latitude'] for site in sites]
    lons = [site['longitude'] for site in sites]
    center_lat = np.mean(lats) if lats else -25.2744
    center_lon = np.mean(lons) if lons else 133.7751

    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

    # Add markers for each site
    for site in sites:
        risk_level = site.get('risk_level', 'MINIMAL')
        probability = site.get('rockfall_probability', 0)

        # Determine marker color and size
        color = get_risk_color(risk_level).replace('#', '')
        radius = max(8, min(25, probability * 30))

        # Create popup with live data
        popup_html = f"""
        <div style="width: 250px;">
            <h4>{site['site_id']}</h4>
            <hr>
            <b>Risk Level:</b> <span style="color: #{color}; font-weight: bold;">{risk_level}</span><br>
            <b>Probability:</b> {probability:.1%}<br>
            <b>Confidence:</b> {site.get('confidence', 0):.1%}<br>
            <hr>
            <b>Live Conditions:</b><br>
            🌡️ Temperature: {site.get('temperature', 0):.1f}°C<br>
            💧 Humidity: {site.get('humidity', 0):.1f}%<br>
            🌧️ Rainfall: {site.get('rainfall', 0):.1f}mm<br>
            📳 Vibration: {site.get('vibration_intensity', 0):.1f}<br>
            <hr>
            <small>Last Update: {site['timestamp'][:16]}</small>
        </div>
        """
        # Add pulsing effect for high risk sites
        if risk_level in ['CRITICAL', 'HIGH']:
            folium.CircleMarker(
                location=[site['latitude'], site['longitude']],
                radius=radius + 5,
                popup=folium.Popup(popup_html, max_width=300),
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.3,
                weight=3,
                opacity=0.8
            ).add_to(m)
        
        # Main marker
        folium.CircleMarker(
            location=[site['latitude'], site['longitude']],
            radius=radius,
            popup=folium.Popup(popup_html, max_width=300),
            color=f"#{color}",
            fill=True,
            fillColor=f"#{color}",
            fillOpacity=0.8,
            weight=2
        ).add_to(m)
        
        # Add site label
        folium.Marker(
            location=[site['latitude'], site['longitude']],
            icon=folium.DivIcon(
                html=f'<div style="background: white; border: 1px solid #{color}; border-radius: 3px; padding: 2px 4px; font-size: 10px; font-weight: bold;">{site["site_id"].split("_")[0]}</div>',
                class_name='site-label'
            )
        ).add_to(m)
    return m

def create_live_time_series():
    """Create live time series chart with recent data"""
    sites_data = make_api_request("/live/sites")
    if not sites_data or not sites_data['sites']:
        return go.Figure().add_annotation(text="No live data available", 
                                          xref="paper", yref="paper", x=0.5, y=0.5)
    
    # Get historical data for time series
    all_sensor_data = []
    all_prediction_data = []
    
    for site in sites_data['sites'][:3]:  # Limit to first 3 sites for performance
        site_id = site['site_id']
        sensor_data = make_api_request(f"/sensors/{site_id}?limit=50")
        prediction_data = make_api_request(f"/predictions/{site_id}?limit=50")
        
        if sensor_data and sensor_data['data']:
            for record in sensor_data['data']:
                record['site_id'] = site_id
                all_sensor_data.extend([record])
        
        if prediction_data and prediction_data['predictions']:
            for record in prediction_data['predictions']:
                record['site_id'] = site_id
                all_prediction_data.extend([record])
    
    if not all_sensor_data:
        return go.Figure().add_annotation(text="No sensor data available", 
                                          xref="paper", yref="paper", x=0.5, y=0.5)
    
    df_sensors = pd.DataFrame(all_sensor_data)
    df_sensors['timestamp'] = pd.to_datetime(df_sensors['timestamp'])
    df_sensors = df_sensors.sort_values('timestamp')
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('🌡️ Temperature Trends', '💧 Humidity & Rainfall', '📳 Vibration Levels',
                        '⛰️ Rockfall Risk Timeline', '🎯 Risk Distribution', '📊 Live Site Status'),
        specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "pie"}, {"type": "bar"}]]
    )
    
    # Temperature trends by site
    sites = df_sensors['site_id'].unique()
    colors = px.colors.qualitative.Set1
    for i, site in enumerate(sites[:3]):
        site_data = df_sensors[df_sensors['site_id'] == site]
        fig.add_trace(
            go.Scatter(x=site_data['timestamp'], y=site_data['temperature'], 
                       name=f'{site} Temp', line=dict(color=colors[i % len(colors)])),
            row=1, col=1
        )
    
    # Humidity and Rainfall
    for i, site in enumerate(sites[:2]):
        site_data = df_sensors[df_sensors['site_id'] == site]
        fig.add_trace(
            go.Scatter(x=site_data['timestamp'], y=site_data['humidity'], 
                       name=f'{site} Humidity', line=dict(color=colors[i])),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=site_data['timestamp'], y=site_data['rainfall'], 
                   name=f'{site} Rainfall', marker_color=colors[i], opacity=0.6),
            row=1, col=2, secondary_y=True
        )
    
    # Vibration levels
    for i, site in enumerate(sites[:3]):
        site_data = df_sensors[df_sensors['site_id'] == site]
        fig.add_trace(
            go.Scatter(
                x=site_data['timestamp'],
                y=site_data['vibration_intensity'],
                name=f'{site} Vibration',
                line=dict(color=colors[i % len(colors)], width=3)
            ),
            row=1, col=3
        )

    # Risk timeline
    if all_prediction_data:
        df_pred = pd.DataFrame(all_prediction_data)
        df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'])
        df_pred = df_pred.sort_values('timestamp')

        for i, site in enumerate(df_pred['site_id'].unique()[:3]):
            site_pred = df_pred[df_pred['site_id'] == site]
            risk_colors = [get_risk_color(risk) for risk in site_pred['risk_level']]

            fig.add_trace(
                go.Scatter(
                    x=site_pred['timestamp'],
                    y=site_pred['rockfall_probability'],
                    name=f'{site} Risk',
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(color=risk_colors, size=8)
                ),
                row=2, col=1
            )

        # Risk distribution pie chart
        risk_dist = df_pred['risk_level'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=risk_dist.index,
                values=risk_dist.values,
                marker=dict(colors=[get_risk_color(level) for level in risk_dist.index]),
                name="Risk Distribution"
            ),
            row=2, col=2
        )

    # Live site status bar chart
    live_sites = sites_data['sites']
    site_names = [site['site_id'].split('_')[0] for site in live_sites]
    risk_probs = [site.get('rockfall_probability', 0) for site in live_sites]
    risk_levels = [site.get('risk_level', 'MINIMAL') for site in live_sites]
    bar_colors = [get_risk_color(level) for level in risk_levels]

    fig.add_trace(
        go.Bar(
            x=site_names,
            y=risk_probs,
            name="Current Risk",
            marker_color=bar_colors,
            text=[f"{p:.1%}" for p in risk_probs],
            textposition='auto'
        ),
        row=2, col=3
    )

    # Add risk threshold lines
    fig.add_hline(y=0.8, line_dash="dash", line_color="red",
                  annotation_text="Critical Threshold", row=2, col=1)
    fig.add_hline(y=0.6, line_dash="dash", line_color="orange",
                  annotation_text="High Risk", row=2, col=1)
    fig.add_hline(y=0.8, line_dash="dash", line_color="red", row=2, col=3)
    fig.add_hline(y=0.6, line_dash="dash", line_color="orange", row=2, col=3)

    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="🔴 LIVE Monitoring Dashboard - Real-time Data Streams",
        title_x=0.5,
        title_font_size=20
    )

    # Update axes labels
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_xaxes(title_text="Time", row=1, col=3)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Mining Sites", row=2, col=3)

    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Humidity (%)", row=1, col=2)
    fig.update_yaxes(title_text="Rainfall (mm)", secondary_y=True, row=1, col=2)
    fig.update_yaxes(title_text="Vibration Intensity", row=1, col=3)
    fig.update_yaxes(title_text="Risk Probability", row=2, col=1)
    fig.update_yaxes(title_text="Risk Probability", row=2, col=3)

    return fig

def create_live_alerts_timeline():
    """Create timeline of recent alerts"""
    alerts_data = make_api_request("/alerts/recent?limit=20")
    if not alerts_data or not alerts_data['alerts']:
        return go.Figure().add_annotation(text="No recent alerts", 
                                          xref="paper", yref="paper", x=0.5, y=0.5)
    
    df_alerts = pd.DataFrame(alerts_data['alerts'])
    df_alerts['timestamp'] = pd.to_datetime(df_alerts['timestamp'])
    df_alerts = df_alerts.sort_values('timestamp')
    
    # Create timeline chart
    fig = go.Figure()
    
    # Color map for risk levels
    risk_colors = {
        'CRITICAL': '#d62728',
        'HIGH': '#ff7f0e',
        'MEDIUM': '#ffbb78',
        'LOW': '#2ca02c',
        'MINIMAL': '#98df8a'
    }
    
    for _, alert in df_alerts.iterrows():
        color = risk_colors.get(alert['risk_level'], '#1f77b4')
        fig.add_trace(go.Scatter(
            x=[alert['timestamp']],
            y=[alert['site_id']],
            mode='markers',
            marker=dict(
                size=alert['probability'] * 30 + 10,
                color=color,
                symbol='triangle-up' if alert['risk_level'] in ['CRITICAL', 'HIGH'] else 'circle',
                line=dict(width=2, color='white')
            ),
            name=alert['risk_level'],
            text=f"Risk: {alert['probability']:.1%}<br>Message: {alert['message']}",
            hovertemplate='<b>%{y}</b><br>Time: %{x}<br>%{text}<extra></extra>',
            showlegend=False
        ))
    
    fig.update_layout(
        title="🚨 Live Alerts Timeline",
        xaxis_title="Time",
        yaxis_title="Mining Sites",
        height=400,
        hovermode='closest'
    )
    
    return fig

# Helper functions for the Site Analysis page
def create_risk_gauge(probability, risk_level):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Rockfall Probability", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': get_risk_color(risk_level)},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': get_risk_color('MINIMAL')},
                {'range': [30, 60], 'color': get_risk_color('LOW')},
                {'range': [60, 80], 'color': get_risk_color('MEDIUM')},
                {'range': [80, 95], 'color': get_risk_color('HIGH')},
                {'range': [95, 100], 'color': get_risk_color('CRITICAL')}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }))
    fig.update_layout(height=250)
    return fig

def create_contributing_factors_chart(factors):
    if not factors:
        return go.Figure().add_annotation(text="No contributing factors data", x=0.5, y=0.5, showarrow=False)
    
    factor_names = list(factors.keys())
    factor_values = list(factors.values())
    
    fig = px.bar(
        x=factor_values,
        y=factor_names,
        orientation='h',
        title='Contributing Factors',
        labels={'x': 'Impact Score', 'y': 'Factor'},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(xaxis_title="Impact Score (Normalized)")
    return fig

def create_time_series_chart(sensor_data, predictions_data):
    df_sensors = pd.DataFrame(sensor_data)
    df_sensors['timestamp'] = pd.to_datetime(df_sensors['timestamp'])
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add sensor data
    fig.add_trace(go.Scatter(x=df_sensors['timestamp'], y=df_sensors['temperature'], name='Temperature (°C)'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_sensors['timestamp'], y=df_sensors['rainfall'], name='Rainfall (mm)'), secondary_y=True)
    fig.add_trace(go.Scatter(x=df_sensors['timestamp'], y=df_sensors['vibration_intensity'], name='Vibration'), secondary_y=False)
    
    # Add predictions
    if predictions_data:
        df_pred = pd.DataFrame(predictions_data)
        df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'])
        fig.add_trace(go.Scatter(x=df_pred['timestamp'], y=df_pred['rockfall_probability'], name='Risk Probability', line=dict(dash='dot', color='red', width=3)), secondary_y=False)

    fig.update_layout(title_text="Historical Sensor Data and Risk Predictions", height=600)
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Sensor Value", secondary_y=False)
    fig.update_yaxes(title_text="Rainfall (mm)", secondary_y=True)

    return fig

# Main Dashboard
def main():
    st.title("⛰️ Rockfall Prediction Dashboard")
    st.markdown("AI-powered real-time monitoring system for open-pit mines")
    
    # Live alert banner (always at top)
    create_live_alert_banner()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Controls")
        
        # Auto refresh toggle
        auto_refresh = st.checkbox("🔄 Auto-refresh (30s)", value=True)
        refresh_interval = st.slider("Refresh interval (seconds)", 10, 120, 30)
        
        if st.button("🔄 Manual Refresh"):
            st.session_state.last_update = datetime.now()
            if auto_refresh:
                time.sleep(1)
                st.rerun()
        
        st.header("📊 Navigation")
        page = st.selectbox("Select Page", 
                            ["🔴 Live Overview", "📍 Site Analysis", "📤 Data Upload", 
                             "⚠️ Alert Center", "🔧 System Health"])
        
        # Live system status in sidebar
        st.header("🟢 System Status")
        health_data = make_api_request("/health")
        if health_data:
            st.success("✅ System Online")
            st.info(f"🤖 Model: {'✅' if health_data['model_loaded'] else '❌'}")
            st.info(f"🗄️ Database: {'✅' if health_data['database_connected'] else '❌'}")
        else:
            st.error("❌ System Offline")
    
    # Get sites data for all pages
    sites_data = make_api_request("/sites")
    available_sites = []
    if sites_data:
        available_sites = [site['site_id'] for site in sites_data['sites']]
    
    if page == "🔴 Live Overview":
        st.header("🔴 LIVE System Overview")
        
        # Live statistics cards
        create_live_stats_cards()
        
        # Two column layout for main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🗺️ Live Risk Heatmap")
            risk_map = create_risk_heatmap_live()
            if risk_map:
                st_folium(risk_map, width=700, height=500, key="live_map")
            
            st.subheader("📊 Real-time Data Streams")
            live_chart = create_live_time_series()
            st.plotly_chart(live_chart, use_container_width=True)
        
        with col2:
            st.subheader("⚠️ Live Alerts Feed")
            alerts_timeline = create_live_alerts_timeline()
            st.plotly_chart(alerts_timeline, use_container_width=True)
            
            # Recent alerts list
            st.subheader("📋 Recent Alerts")
            alerts_data = make_api_request("/alerts/recent?limit=10")
            if alerts_data and alerts_data['alerts']:
                for alert in alerts_data['alerts'][:5]:
                    risk_class = alert['risk_level'].lower()
                    time_ago = (datetime.now() - datetime.fromisoformat(alert['timestamp'].replace('Z', '+00:00'))).total_seconds() / 60
                    st.markdown(f"""
                    <div class="metric-card {risk_class}">
                        <strong>{alert['site_id']}</strong> - {alert['risk_level']}<br>
                        <small>Risk: {alert['probability']:.1%} | {time_ago:.0f}m ago</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No recent alerts - All systems normal")
        
        # Auto-refresh functionality
        if auto_refresh:
            placeholder = st.empty()
            with placeholder.container():
                st.info(f"🔄 Auto-refreshing every {refresh_interval} seconds...")
                time.sleep(refresh_interval)
                st.rerun()
    
    elif page == "📍 Site Analysis":
        st.header("📍 Detailed Site Analysis")
        
        if not available_sites:
            st.warning("No sites available. System may be starting up...")
            return
        
        selected_site = st.selectbox("🏭 Select Mining Site", available_sites)
        
        if selected_site:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Get latest data for selected site
                predictions_data = make_api_request(f"/predictions/{selected_site}?limit=10")
                sensor_data = make_api_request(f"/sensors/{selected_site}?limit=50")
                
                if predictions_data and predictions_data['predictions']:
                    latest_prediction = predictions_data['predictions'][0]
                    
                    st.subheader("🎯 Current Risk Status")
                    risk_level = latest_prediction['risk_level']
                    probability = latest_prediction['rockfall_probability']
                    confidence = latest_prediction['confidence']
                    
                    # Risk gauge
                    risk_gauge = create_risk_gauge(probability, risk_level)
                    st.plotly_chart(risk_gauge, use_container_width=True)
                    
                    # Status card
                    risk_class = risk_level.lower()
                    st.markdown(f"""
                    <div class="metric-card {risk_class}">
                        <h3>🏭 {selected_site}</h3>
                        <hr>
                        <p><strong>🎯 Risk Level:</strong> {risk_level}</p>
                        <p><strong>📊 Probability:</strong> {probability:.1%}</p>
                        <p><strong>🔍 Confidence:</strong> {confidence:.1%}</p>
                        <p><strong>🕐 Last Update:</strong> {latest_prediction['timestamp'][:16]}</p>
                        <hr>
                        <small>Live monitoring active ✅</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Contributing factors
                    st.subheader("📈 Contributing Factors")
                    factors_chart = create_contributing_factors_chart(latest_prediction['contributing_factors'])
                    st.plotly_chart(factors_chart, use_container_width=True)
                    
                    # Recommendations
                    st.subheader("⚡ Recommended Actions")
                    for i, rec in enumerate(latest_prediction['recommendations'], 1):
                        priority = "🔴" if i <= 2 else "🟡" if i <= 4 else "🟢"
                        st.markdown(f"{priority} **{i}.** {rec}")
            
            with col2:
                st.subheader("📊 Historical Trends & Live Data")
                if sensor_data and sensor_data['data']:
                    time_series_chart = create_time_series_chart(
                        sensor_data['data'], predictions_data['predictions']
                    )
                    st.plotly_chart(time_series_chart, use_container_width=True)
                else:
                    st.info("No historical sensor data available for this site")
                
                # Prediction history table
                st.subheader("📋 Recent Predictions")
                if predictions_data and predictions_data['predictions']:
                    pred_df = pd.DataFrame(predictions_data['predictions'])
                    pred_df['probability_pct'] = pred_df['rockfall_probability'].apply(lambda x: f"{x:.1%}")
                    pred_df['time'] = pd.to_datetime(pred_df['timestamp']).dt.strftime('%H:%M')
                    
                    # Color code the dataframe
                    def color_risk(val):
                        colors = {
                            'CRITICAL': 'background-color: #ffcccb',
                            'HIGH': 'background-color: #ffe4b5',
                            'MEDIUM': 'background-color: #fff8dc',
                            'LOW': 'background-color: #f0fff0',
                            'MINIMAL': 'background-color: #f8fff8'
                        }
                        return colors.get(val, '')
                    
                    styled_df = pred_df[['time', 'risk_level', 'probability_pct', 'confidence']].style.applymap(
                        color_risk, subset=['risk_level']
                    )
                    st.dataframe(styled_df, use_container_width=True)
    
    elif page == "📤 Data Upload":
        st.header("📤 Data Upload and Management")
        
        tab1, tab2 = st.tabs(["📁 File Upload", "✋ Manual Entry"])
        
        with tab1:
            st.subheader("📁 Upload Sensor Data")
            uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
            
            if uploaded_file:
                # Preview data
                df = pd.read_csv(uploaded_file)
                st.subheader("👀 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Total Records", len(df))
                with col2:
                    st.metric("📍 Unique Sites", df['site_id'].nunique() if 'site_id' in df.columns else 0)
                with col3:
                    st.metric("📅 Date Range", f"{len(df)} records")
                
                if st.button("🚀 Upload Data", type="primary"):
                    try:
                        files = {"file": uploaded_file.getvalue()}
                        response = requests.post(f"{API_BASE_URL}/upload/sensors", 
                                                 files={"file": ("data.csv", uploaded_file.getvalue(), "text/csv")})
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"✅ Successfully uploaded {result['count']} records!")
                            st.balloons()
                        else:
                            st.error(f"❌ Upload failed: {response.text}")
                            
                    except Exception as e:
                        st.error(f"❌ Upload error: {str(e)}")
        
        with tab2:
            st.subheader("✋ Manual Data Entry & Live Testing")
            st.warning("This section is not yet implemented.")

    elif page == "⚠️ Alert Center":
        st.header("⚠️ Alert Management and History")
        
        alert_limit = st.slider("Number of recent alerts to show", 10, 100, 25)
        alerts_data = make_api_request(f"/alerts/recent?limit={alert_limit}")
        
        if alerts_data and alerts_data['alerts']:
            df_alerts = pd.DataFrame(alerts_data['alerts'])
            
            # Filter and sort
            risk_filter = st.multiselect("Filter by Risk Level", ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL'], default=['CRITICAL', 'HIGH', 'MEDIUM'])
            
            if risk_filter:
                df_alerts = df_alerts[df_alerts['risk_level'].isin(risk_filter)]
            
            st.markdown(f"**Showing {len(df_alerts)} recent alerts**")
            
            # Display alerts in a table
            st.dataframe(df_alerts.sort_values(by='timestamp', ascending=False), use_container_width=True)
            
            # Download CSV
            csv = df_alerts.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Alerts as CSV",
                data=csv,
                file_name='rockfall_alerts.csv',
                mime='text/csv',
            )
        else:
            st.info("No alerts found for the selected criteria.")
    
    elif page == "🔧 System Health":
        st.header("🔧 System Health and Diagnostics")
        
        st.subheader("API Status")
        health_data = make_api_request("/health")
        if health_data:
            st.json(health_data)
        else:
            st.warning("Could not retrieve API health data.")
        
        st.subheader("Database Health")
        # Assuming you have a /database-health endpoint in your API
        db_health_data = make_api_request("/database-health")
        if db_health_data:
            st.json(db_health_data)
        else:
            st.warning("Could not retrieve database health data.")
            
        st.subheader("Prediction Model Info")
        model_info = make_api_request("/model-info")
        if model_info:
            st.json(model_info)
        else:
            st.warning("Could not retrieve model information.")

if __name__ == "__main__":
    main()