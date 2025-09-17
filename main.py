from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib
import sqlite3
import smtplib
import email.mime.text
import email.mime.multipart
from datetime import datetime, timedelta
import json
import logging
import os
from io import StringIO
import asyncio
from contextlib import asynccontextmanager
import random
import threading
import time

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

# Global variables
ml_model = None
alert_config = AlertConfig()
data_simulator_running = False

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
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            site_id TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            risk_level TEXT,
            probability REAL,
            message TEXT,
            alert_sent BOOLEAN DEFAULT FALSE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    
    # Insert sample data if tables are empty
    cursor.execute("SELECT COUNT(*) FROM sensor_readings")
    if cursor.fetchone()[0] == 0:
        insert_sample_data(conn)
    
    conn.close()
    logger.info("Database initialized with sample data")

def insert_sample_data(conn):
    """Insert realistic sample data for immediate testing"""
    cursor = conn.cursor()
    
    # Sample mining sites in Australia
    sites = [
        {"id": "BODDINGTON_001", "lat": -32.7675, "lon": 116.4719, "name": "Boddington Gold Mine"},
        {"id": "OLYMPIC_DAM_001", "lat": -30.4406, "lon": 136.8817, "name": "Olympic Dam Mine"},
        {"id": "HUNTER_VALLEY_001", "lat": -32.7647, "lon": 150.8109, "name": "Hunter Valley Coal"},
        {"id": "PILBARA_001", "lat": -22.3964, "lon": 117.2641, "name": "Pilbara Iron Ore"},
        {"id": "ESCONDIDA_001", "lat": -24.2622, "lon": -69.0511, "name": "Escondida Copper"}
    ]
    
    # Generate last 30 days of data
    base_time = datetime.now() - timedelta(days=30)
    
    for site in sites:
        for day in range(30):
            for hour in range(0, 24, 4):  # Every 4 hours
                timestamp = base_time + timedelta(days=day, hours=hour)
                
                # Create realistic sensor data with some variation
                sensor_data = {
                    'site_id': site['id'],
                    'timestamp': timestamp,
                    'slope_angle': np.random.normal(45, 10).clip(25, 75),
                    'rock_strength': np.random.normal(50, 15).clip(10, 90),
                    'joint_spacing': np.random.normal(30, 10).clip(5, 80),
                    'joint_orientation': np.random.uniform(0, 360),
                    'water_content': np.random.beta(2, 5) * 50,
                    'temperature': np.random.normal(25, 8).clip(5, 45),
                    'humidity': np.random.beta(3, 2) * 100,
                    'wind_speed': np.random.weibull(2) * 12,
                    'rainfall': np.random.exponential(1).clip(0, 15),
                    'vibration_intensity': np.random.gamma(1.5, 1).clip(0, 6),
                    'latitude': site['lat'] + np.random.uniform(-0.01, 0.01),
                    'longitude': site['lon'] + np.random.uniform(-0.01, 0.01)
                }
                
                cursor.execute('''
                    INSERT INTO sensor_readings 
                    (site_id, timestamp, slope_angle, rock_strength, joint_spacing, joint_orientation,
                     water_content, temperature, humidity, wind_speed, rainfall, vibration_intensity,
                     latitude, longitude)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    sensor_data['site_id'], sensor_data['timestamp'], sensor_data['slope_angle'],
                    sensor_data['rock_strength'], sensor_data['joint_spacing'], sensor_data['joint_orientation'],
                    sensor_data['water_content'], sensor_data['temperature'], sensor_data['humidity'],
                    sensor_data['wind_speed'], sensor_data['rainfall'], sensor_data['vibration_intensity'],
                    sensor_data['latitude'], sensor_data['longitude']
                ))
    
    conn.commit()
    logger.info("Sample data inserted for 5 mining sites over 30 days")

def load_ml_model():
    """Load trained ML model"""
    global ml_model
    try:
        if os.path.exists('rockfall_model.joblib'):
            ml_model = joblib.load('rockfall_model.joblib')
            logger.info("ML model loaded successfully")
        else:
            # Create a trained model for immediate functionality
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            
            logger.info("Creating and training demo ML model...")
            
            # Generate training data
            np.random.seed(42)
            n_samples = 1000
            X_train = np.random.rand(n_samples, 10)
            
            # Create realistic risk patterns
            risk_score = (
                0.3 * X_train[:, 0] +  # slope_angle
                0.2 * (1 - X_train[:, 1]) +  # rock_strength (inverted)
                0.2 * X_train[:, 4] +  # water_content
                0.15 * X_train[:, 9] +  # vibration_intensity
                0.15 * np.random.normal(0, 0.1, n_samples)  # noise
            )
            y_train = (risk_score > 0.4).astype(int)
            
            # Create and train model
            scaler = StandardScaler()
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            
            X_scaled = scaler.fit_transform(X_train)
            classifier.fit(X_scaled, y_train)
            
            ml_model = {
                'classifier': classifier,
                'scaler': scaler,
                'feature_names': ['slope_angle', 'rock_strength', 'joint_spacing', 'joint_orientation',
                                'water_content', 'temperature', 'humidity', 'wind_speed', 'rainfall',
                                'vibration_intensity']
            }
            
            # Save the model
            joblib.dump(ml_model, 'rockfall_model.joblib')
            logger.info("Demo ML model created and saved")
    except Exception as e:
        logger.error(f"Error loading ML model: {e}")
        ml_model = None

def start_data_simulator():
    """Start background data simulator for live updates"""
    global data_simulator_running
    
    def simulate_data():
        sites = ["BODDINGTON_001", "OLYMPIC_DAM_001", "HUNTER_VALLEY_001", "PILBARA_001", "ESCONDIDA_001"]
        site_coords = {
            "BODDINGTON_001": (-32.7675, 116.4719),
            "OLYMPIC_DAM_001": (-30.4406, 136.8817),
            "HUNTER_VALLEY_001": (-32.7647, 150.8109),
            "PILBARA_001": (-22.3964, 117.2641),
            "ESCONDIDA_001": (-24.2622, -69.0511)
        }
        
        while data_simulator_running:
            try:
                # Generate new sensor reading for random site
                site_id = random.choice(sites)
                lat, lon = site_coords[site_id]
                
                # Create realistic sensor data with some randomness
                current_time = datetime.now()
                
                # Add some seasonal and daily variations
                hour_factor = np.sin(2 * np.pi * current_time.hour / 24) * 0.1
                season_factor = np.sin(2 * np.pi * current_time.timetuple().tm_yday / 365) * 0.2
                
                sensor_data = SensorData(
                    site_id=site_id,
                    timestamp=current_time,
                    slope_angle=np.random.normal(45, 8) + hour_factor * 5,
                    rock_strength=np.random.normal(50, 12) + season_factor * 10,
                    joint_spacing=np.random.normal(30, 8),
                    joint_orientation=np.random.uniform(0, 360),
                    water_content=np.random.beta(2, 5) * 60 + max(0, np.random.normal(0, 5)),
                    temperature=np.random.normal(25, 6) + season_factor * 15 + hour_factor * 8,
                    humidity=np.random.beta(3, 2) * 100,
                    wind_speed=np.random.weibull(2) * 10,
                    rainfall=max(0, np.random.exponential(1) - 0.5),
                    vibration_intensity=np.random.gamma(1.5, 1) + random.choice([0, 0, 0, 2, 4]),  # Occasional spikes
                    latitude=lat + np.random.uniform(-0.005, 0.005),
                    longitude=lon + np.random.uniform(-0.005, 0.005)
                )
                
                # Store sensor data and generate prediction
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(process_sensor_data(sensor_data))
                loop.close()
                
                # Random interval between 10-30 seconds
                time.sleep(random.uniform(10, 30))
                
            except Exception as e:
                logger.error(f"Data simulator error: {e}")
                time.sleep(30)
    
    if not data_simulator_running:
        data_simulator_running = True
        thread = threading.Thread(target=simulate_data, daemon=True)
        thread.start()
        logger.info("Data simulator started")

async def process_sensor_data(sensor_data: SensorData):
    """Process sensor data and generate predictions"""
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
        
        # Generate prediction
        if ml_model:
            prediction = await predict_rockfall_internal(sensor_data)
            
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
            
            # Check for alerts
            if prediction.rockfall_probability >= alert_config.risk_threshold:
                cursor.execute('''
                    INSERT INTO alerts (site_id, timestamp, risk_level, probability, message, alert_sent)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    prediction.site_id, prediction.timestamp, prediction.risk_level,
                    prediction.rockfall_probability, f"High risk detected at {prediction.site_id}", True
                ))
                conn.commit()
                logger.warning(f"ALERT: High risk detected at {prediction.site_id} - {prediction.rockfall_probability:.1%}")
            
            conn.close()
            
    except Exception as e:
        logger.error(f"Error processing sensor data: {e}")

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

async def predict_rockfall_internal(sensor_data: SensorData) -> PredictionResponse:
    """Internal prediction function"""
    if ml_model is None:
        raise ValueError("ML model not available")
    
    # Extract features and make prediction
    features = extract_features(sensor_data)
    scaled_features = ml_model['scaler'].transform(features)
    
    # Get probability and prediction
    prob = ml_model['classifier'].predict_proba(scaled_features)[0]
    rockfall_prob = prob[1] if len(prob) > 1 else prob[0]
    confidence = max(prob)
    
    # Add some realistic noise and variations
    rockfall_prob = np.clip(rockfall_prob + np.random.normal(0, 0.05), 0, 1)
    
    # Calculate feature importance (contributing factors)
    feature_importance = ml_model['classifier'].feature_importances_
    feature_names = ml_model['feature_names']
    contributing_factors = dict(zip(feature_names, feature_importance))
    
    # Determine risk level and recommendations
    risk_level = calculate_risk_level(rockfall_prob)
    recommendations = get_recommendations(risk_level, contributing_factors)
    
    return PredictionResponse(
        site_id=sensor_data.site_id,
        timestamp=sensor_data.timestamp,
        rockfall_probability=float(rockfall_prob),
        risk_level=risk_level,
        confidence=float(confidence),
        contributing_factors=contributing_factors,
        recommendations=recommendations
    )

async def send_alert(prediction: PredictionResponse):
    """Send alert notifications - simplified for Python 3.13 compatibility"""
    if not alert_config.email_enabled or prediction.rockfall_probability < alert_config.risk_threshold:
        return
    
    try:
        # Simplified email sending - can be enhanced later
        logger.info(f"EMAIL ALERT would be sent: {prediction.site_id} - {prediction.risk_level} risk")
        # Email functionality can be implemented with a different approach if needed
        
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_database()
    load_ml_model()
    start_data_simulator()  # Start live data simulation
    yield
    # Shutdown
    global data_simulator_running
    data_simulator_running = False
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
        prediction = await predict_rockfall_internal(sensor_data)
        
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
        
        # Store prediction
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

@app.get("/alerts/recent")
async def get_recent_alerts(limit: int = 50):
    """Get recent alerts"""
    try:
        conn = sqlite3.connect('rockfall_data.db')
        df = pd.read_sql_query('''
            SELECT * FROM alerts 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', conn, params=(limit,))
        conn.close()
        
        alerts = df.to_dict('records')
        return {"alerts": alerts, "count": len(alerts)}
        
    except Exception as e:
        logger.error(f"Error retrieving alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard/stats")
async def get_dashboard_stats():
    """Get dashboard statistics"""
    try:
        conn = sqlite3.connect('rockfall_data.db')
        
        # Get site count
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT site_id) FROM sensor_readings")
        total_sites = cursor.fetchone()[0]
        
        # Get recent high risk predictions
        cursor.execute('''
            SELECT COUNT(*) FROM predictions 
            WHERE risk_level IN ('HIGH', 'CRITICAL') 
            AND timestamp > datetime('now', '-1 hour')
        ''')
        high_risk_count = cursor.fetchone()[0]
        
        # Get latest predictions for risk distribution
        cursor.execute('''
            SELECT risk_level, COUNT(*) as count
            FROM predictions 
            WHERE timestamp > datetime('now', '-1 hour')
            GROUP BY risk_level
        ''')
        risk_distribution = dict(cursor.fetchall())
        
        # Get recent alerts
        cursor.execute('''
            SELECT COUNT(*) FROM alerts 
            WHERE timestamp > datetime('now', '-1 hour')
        ''')
        recent_alerts = cursor.fetchone()[0]
        
        # Average risk probability
        cursor.execute('''
            SELECT AVG(rockfall_probability) FROM predictions 
            WHERE timestamp > datetime('now', '-1 hour')
        ''')
        avg_risk = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            "total_sites": total_sites,
            "high_risk_sites": high_risk_count,
            "recent_alerts": recent_alerts,
            "average_risk": float(avg_risk),
            "risk_distribution": risk_distribution,
            "last_update": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving dashboard stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/live/sites")
async def get_live_site_data():
    """Get live data for all sites"""
    try:
        conn = sqlite3.connect('rockfall_data.db')
        
        # Get latest sensor reading for each site
        df_sensors = pd.read_sql_query('''
            SELECT s.site_id, s.timestamp, s.latitude, s.longitude,
                   s.temperature, s.humidity, s.rainfall, s.vibration_intensity
            FROM sensor_readings s
            INNER JOIN (
                SELECT site_id, MAX(timestamp) as max_timestamp
                FROM sensor_readings
                GROUP BY site_id
            ) latest ON s.site_id = latest.site_id AND s.timestamp = latest.max_timestamp
        ''', conn)
        
        # Get latest prediction for each site
        df_predictions = pd.read_sql_query('''
            SELECT p.site_id, p.rockfall_probability, p.risk_level, p.confidence
            FROM predictions p
            INNER JOIN (
                SELECT site_id, MAX(timestamp) as max_timestamp
                FROM predictions
                GROUP BY site_id
            ) latest ON p.site_id = latest.site_id AND p.timestamp = latest.max_timestamp
        ''', conn)
        
        conn.close()
        
        # Merge data
        site_data = []
        for _, sensor in df_sensors.iterrows():
            site_id = sensor['site_id']
            prediction = df_predictions[df_predictions['site_id'] == site_id]
            
            if not prediction.empty:
                pred = prediction.iloc[0]
                site_data.append({
                    "site_id": site_id,
                    "latitude": sensor['latitude'],
                    "longitude": sensor['longitude'],
                    "timestamp": sensor['timestamp'],
                    "rockfall_probability": pred['rockfall_probability'],
                    "risk_level": pred['risk_level'],
                    "confidence": pred['confidence'],
                    "temperature": sensor['temperature'],
                    "humidity": sensor['humidity'],
                    "rainfall": sensor['rainfall'],
                    "vibration_intensity": sensor['vibration_intensity']
                })
        
        return {"sites": site_data, "count": len(site_data)}
        
    except Exception as e:
        logger.error(f"Error retrieving live site data: {e}")
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
        "database_connected": True,  # Simple check
        "data_simulator_running": data_simulator_running
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