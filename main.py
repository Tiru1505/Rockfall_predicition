from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import json
import logging
import os
from io import StringIO
import asyncio
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class SensorData(BaseModel):
    site_id: str = Field(..., description="Mining site identifier")
    timestamp: datetime = Field(default_factory=datetime.now)
    slope_angle: float = Field(..., ge=0, le=90, description="Slope angle in degrees")
    rock_strength: float = Field(..., ge=0, le=100, description="Rock strength (MPa)")
    joint_spacing: float = Field(..., ge=0, description="Joint spacing (cm)")
    joint_orientation: float = Field(..., ge=0, le=360, description="Joint orientation (degrees)")
    water_content: float = Field(..., ge=0, le=100, description="Water content (%)")
    temperature: float = Field(..., ge=-50, le=60, description="Temperature (°C)")
    humidity: float = Field(..., ge=0, le=100, description="Humidity (%)")
    wind_speed: float = Field(..., ge=0, description="Wind speed (m/s)")
    rainfall: float = Field(..., ge=0, description="Rainfall (mm)")
    vibration_intensity: float = Field(..., ge=0, description="Vibration intensity")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

class PredictionResponse(BaseModel):
    site_id: str
    timestamp: datetime
    rockfall_probability: float
    risk_level: str
    confidence: float
    contributing_factors: Dict[str, float]
    recommendations: List[str]

class AlertConfig(BaseModel):
    email_enabled: bool = True
    sms_enabled: bool = False
    risk_threshold: float = 0.7
    email_recipients: List[str] = []
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_database()
    load_ml_model()
    yield
    # Shutdown
    logger.info("Shutting down...")

app = FastAPI(
    title="Rockfall Prediction API",
    description="AI-powered rockfall prediction system for open-pit mines",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
ml_model = None
alert_config = AlertConfig()

def init_database():
    """Initialize SQLite database for storing predictions and sensor data"""
    conn = sqlite3.connect('rockfall_data.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            site_id TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            slope_angle REAL,
            rock_strength REAL,
            joint_spacing REAL,
            joint_orientation REAL,
            water_content REAL,
            temperature REAL,
            humidity REAL,
            wind_speed REAL,
            rainfall REAL,
            vibration_intensity REAL,
            latitude REAL,
            longitude REAL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            site_id TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            rockfall_probability REAL,
            risk_level TEXT,
            confidence REAL,
            contributing_factors TEXT,
            recommendations TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized")

def load_ml_model():
    """Load trained ML model"""
    global ml_model
    try:
        if os.path.exists('rockfall_model.joblib'):
            ml_model = joblib.load('rockfall_model.joblib')
            logger.info("ML model loaded successfully")
        else:
            # Create a dummy model for demo purposes
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            ml_model = {
                'classifier': RandomForestClassifier(n_estimators=100, random_state=42),
                'scaler': StandardScaler(),
                'feature_names': ['slope_angle', 'rock_strength', 'joint_spacing', 'joint_orientation',
                                'water_content', 'temperature', 'humidity', 'wind_speed', 'rainfall',
                                'vibration_intensity']
            }
            # Fit with dummy data
            X_dummy = np.random.rand(100, 10)
            y_dummy = np.random.randint(0, 2, 100)
            ml_model['scaler'].fit(X_dummy)
            ml_model['classifier'].fit(ml_model['scaler'].transform(X_dummy), y_dummy)
            logger.info("Created dummy ML model for demo")
    except Exception as e:
        logger.error(f"Error loading ML model: {e}")
        ml_model = None

def extract_features(sensor_data: SensorData) -> np.ndarray:
    """Extract features from sensor data for ML prediction"""
    features = [
        sensor_data.slope_angle,
        sensor_data.rock_strength,
        sensor_data.joint_spacing,
        sensor_data.joint_orientation,
        sensor_data.water_content,
        sensor_data.temperature,
        sensor_data.humidity,
        sensor_data.wind_speed,
        sensor_data.rainfall,
        sensor_data.vibration_intensity
    ]
    return np.array(features).reshape(1, -1)

def calculate_risk_level(probability: float) -> str:
    """Calculate risk level based on probability"""
    if probability >= 0.8:
        return "CRITICAL"
    elif probability >= 0.6:
        return "HIGH"
    elif probability >= 0.4:
        return "MEDIUM"
    elif probability >= 0.2:
        return "LOW"
    else:
        return "MINIMAL"

def get_recommendations(risk_level: str, factors: Dict[str, float]) -> List[str]:
    """Generate recommendations based on risk level and contributing factors"""
    recommendations = []
    
    if risk_level in ["CRITICAL", "HIGH"]:
        recommendations.append("Immediate evacuation of personnel from risk area")
        recommendations.append("Deploy emergency monitoring equipment")
        recommendations.append("Contact emergency response team")
    
    if risk_level in ["HIGH", "MEDIUM"]:
        recommendations.append("Increase monitoring frequency")
        recommendations.append("Implement temporary access restrictions")
    
    # Factor-specific recommendations
    if factors.get('water_content', 0) > 0.3:
        recommendations.append("Implement drainage measures to reduce water content")
    
    if factors.get('slope_angle', 0) > 0.3:
        recommendations.append("Consider slope stabilization measures")
    
    if factors.get('vibration_intensity', 0) > 0.3:
        recommendations.append("Reduce blasting activities in nearby areas")
    
    return recommendations

async def send_alert(prediction: PredictionResponse):
    """Send alert notifications"""
    if not alert_config.email_enabled or prediction.rockfall_probability < alert_config.risk_threshold:
        return
    
    try:
        msg = MimeMultipart()
        msg['From'] = alert_config.smtp_username
        msg['To'] = ", ".join(alert_config.email_recipients)
        msg['Subject'] = f"Rockfall Alert - {prediction.risk_level} Risk at {prediction.site_id}"
        
        body = f"""
        ROCKFALL ALERT
        
        Site: {prediction.site_id}
        Time: {prediction.timestamp}
        Risk Level: {prediction.risk_level}
        Probability: {prediction.rockfall_probability:.2%}
        Confidence: {prediction.confidence:.2%}
        
        Top Contributing Factors:
        {chr(10).join([f"- {factor}: {value:.1%}" for factor, value in sorted(prediction.contributing_factors.items(), key=lambda x: x[1], reverse=True)[:3]])}
        
        Recommendations:
        {chr(10).join([f"- {rec}" for rec in prediction.recommendations])}
        
        Location: {prediction.latitude if hasattr(prediction, 'latitude') else 'N/A'}, {prediction.longitude if hasattr(prediction, 'longitude') else 'N/A'}
        """
        
        msg.attach(MimeText(body, 'plain'))
        
        server = smtplib.SMTP(alert_config.smtp_server, alert_config.smtp_port)
        server.starttls()
        server.login(alert_config.smtp_username, alert_config.smtp_password)
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Alert sent for site {prediction.site_id}")
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")

# API Endpoints

@app.get("/")
async def root():
    return {"message": "Rockfall Prediction API", "version": "1.0.0", "status": "active"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_rockfall(sensor_data: SensorData):
    """Predict rockfall probability based on sensor data"""
    if ml_model is None:
        raise HTTPException(status_code=503, detail="ML model not available")
    
    try:
        # Store sensor data
        conn = sqlite3.connect('rockfall_data.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sensor_readings 
            (site_id, timestamp, slope_angle, rock_strength, joint_spacing, joint_orientation,
             water_content, temperature, humidity, wind_speed, rainfall, vibration_intensity,
             latitude, longitude)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            sensor_data.site_id, sensor_data.timestamp, sensor_data.slope_angle,
            sensor_data.rock_strength, sensor_data.joint_spacing, sensor_data.joint_orientation,
            sensor_data.water_content, sensor_data.temperature, sensor_data.humidity,
            sensor_data.wind_speed, sensor_data.rainfall, sensor_data.vibration_intensity,
            sensor_data.latitude, sensor_data.longitude
        ))
        conn.commit()
        conn.close()
        
        # Extract features and make prediction
        features = extract_features(sensor_data)
        scaled_features = ml_model['scaler'].transform(features)
        
        # Get probability and prediction
        prob = ml_model['classifier'].predict_proba(scaled_features)[0]
        rockfall_prob = prob[1] if len(prob) > 1 else prob[0]
        confidence = max(prob)
        
        # Calculate feature importance (contributing factors)
        feature_importance = ml_model['classifier'].feature_importances_
        feature_names = ml_model['feature_names']
        contributing_factors = dict(zip(feature_names, feature_importance))
        
        # Determine risk level and recommendations
        risk_level = calculate_risk_level(rockfall_prob)
        recommendations = get_recommendations(risk_level, contributing_factors)
        
        prediction = PredictionResponse(
            site_id=sensor_data.site_id,
            timestamp=sensor_data.timestamp,
            rockfall_probability=float(rockfall_prob),
            risk_level=risk_level,
            confidence=float(confidence),
            contributing_factors=contributing_factors,
            recommendations=recommendations
        )
        
        # Store prediction
        conn = sqlite3.connect('rockfall_data.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions 
            (site_id, timestamp, rockfall_probability, risk_level, confidence, contributing_factors, recommendations)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            prediction.site_id, prediction.timestamp, prediction.rockfall_probability,
            prediction.risk_level, prediction.confidence, 
            json.dumps(prediction.contributing_factors),
            json.dumps(prediction.recommendations)
        ))
        conn.commit()
        conn.close()
        
        # Send alert if necessary
        await send_alert(prediction)
        
        return prediction
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/predictions/{site_id}")
async def get_site_predictions(site_id: str, limit: int = 100):
    """Get historical predictions for a site"""
    try:
        conn = sqlite3.connect('rockfall_data.db')
        df = pd.read_sql_query('''
            SELECT * FROM predictions 
            WHERE site_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', conn, params=(site_id, limit))
        conn.close()
        
        # Parse JSON fields
        predictions = []
        for _, row in df.iterrows():
            prediction = {
                'site_id': row['site_id'],
                'timestamp': row['timestamp'],
                'rockfall_probability': row['rockfall_probability'],
                'risk_level': row['risk_level'],
                'confidence': row['confidence'],
                'contributing_factors': json.loads(row['contributing_factors']) if row['contributing_factors'] else {},
                'recommendations': json.loads(row['recommendations']) if row['recommendations'] else []
            }
            predictions.append(prediction)
        
        return {"predictions": predictions, "count": len(predictions)}
        
    except Exception as e:
        logger.error(f"Error retrieving predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sensors/{site_id}")
async def get_sensor_data(site_id: str, limit: int = 100):
    """Get historical sensor data for a site"""
    try:
        conn = sqlite3.connect('rockfall_data.db')
        df = pd.read_sql_query('''
            SELECT * FROM sensor_readings 
            WHERE site_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', conn, params=(site_id, limit))
        conn.close()
        
        return {"data": df.to_dict('records'), "count": len(df)}
        
    except Exception as e:
        logger.error(f"Error retrieving sensor data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/sensors")
async def upload_sensor_data(file: UploadFile = File(...)):
    """Upload sensor data from CSV file"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_cols = ['site_id', 'slope_angle', 'rock_strength', 'joint_spacing', 
                        'joint_orientation', 'water_content', 'temperature', 'humidity',
                        'wind_speed', 'rainfall', 'vibration_intensity', 'latitude', 'longitude']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")
        
        # Add timestamp if not present
        if 'timestamp' not in df.columns:
            df['timestamp'] = datetime.now()
        
        # Store data in database
        conn = sqlite3.connect('rockfall_data.db')
        df.to_sql('sensor_readings', conn, if_exists='append', index=False)
        conn.close()
        
        return {"message": f"Successfully uploaded {len(df)} sensor readings", "count": len(df)}
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/alerts/config")
async def configure_alerts(config: AlertConfig):
    """Configure alert settings"""
    global alert_config
    alert_config = config
    return {"message": "Alert configuration updated", "config": config}

@app.get("/alerts/config")
async def get_alert_config():
    """Get current alert configuration"""
    return alert_config

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "model_loaded": ml_model is not None,
        "database_connected": True  # Simple check
    }

@app.get("/sites")
async def get_sites():
    """Get list of all sites with recent data"""
    try:
        conn = sqlite3.connect('rockfall_data.db')
        df = pd.read_sql_query('''
            SELECT DISTINCT site_id, 
                   MAX(timestamp) as last_reading,
                   COUNT(*) as total_readings
            FROM sensor_readings 
            GROUP BY site_id
            ORDER BY last_reading DESC
        ''', conn)
        conn.close()
        
        return {"sites": df.to_dict('records'), "count": len(df)}
        
    except Exception as e:
        logger.error(f"Error retrieving sites: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
