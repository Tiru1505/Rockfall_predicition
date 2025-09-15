"""
Configuration file for Rockfall Prediction System
Contains all configurable parameters and settings
"""

import os
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import timedelta

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    db_path: str = "rockfall_data.db"
    backup_interval_hours: int = 24
    max_backup_files: int = 7

@dataclass
class ModelConfig:
    """ML Model configuration settings"""
    model_path: str = "rockfall_model.joblib"
    retrain_interval_days: int = 30
    min_samples_for_retrain: int = 1000
    confidence_threshold: float = 0.75
    
    # Risk level thresholds
    risk_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.risk_thresholds is None:
            self.risk_thresholds = {
                "MINIMAL": 0.0,
                "LOW": 0.2, 
                "MEDIUM": 0.4,
                "HIGH": 0.6,
                "CRITICAL": 0.8
            }

@dataclass 
class AlertConfig:
    """Alert system configuration"""
    # Email settings
    email_enabled: bool = True
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = os.getenv("SMTP_USERNAME", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    email_recipients: List[str] = None
    
    # SMS settings (Twilio)
    sms_enabled: bool = False
    twilio_account_sid: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    twilio_auth_token: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    twilio_phone_number: str = os.getenv("TWILIO_PHONE_NUMBER", "")
    sms_recipients: List[str] = None
    
    # Alert thresholds
    critical_threshold: float = 0.8
    high_threshold: float = 0.6
    alert_cooldown_hours: int = 2  # Prevent spam alerts
    
    # Alert templates
    email_subject_template: str = "Rockfall Alert - {risk_level} Risk at {site_id}"
    email_body_template: str = """
ROCKFALL ALERT

Site: {site_id}
Time: {timestamp}
Risk Level: {risk_level}
Probability: {probability:.1%}
Confidence: {confidence:.1%}

Location: {latitude:.4f}, {longitude:.4f}

Top Contributing Factors:
{factors}

Recommended Actions:
{recommendations}

This is an automated alert from the Rockfall Prediction System.
Please take appropriate safety measures immediately.
"""
    
    sms_template: str = "ROCKFALL ALERT: {risk_level} risk at {site_id}. Probability: {probability:.1%}. Take immediate action."
    
    def __post_init__(self):
        if self.email_recipients is None:
            self.email_recipients = []
        if self.sms_recipients is None:
            self.sms_recipients = []

@dataclass
class APIConfig:
    """API configuration settings"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: List[str] = None
    rate_limit_requests: int = 100
    rate_limit_period: str = "1 minute"
    
    # Data validation
    max_upload_size_mb: int = 50
    allowed_file_extensions: List[str] = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]
        if self.allowed_file_extensions is None:
            self.allowed_file_extensions = [".csv", ".xlsx", ".json"]

@dataclass
class DashboardConfig:
    """Streamlit dashboard configuration"""
    page_title: str = "Rockfall Prediction Dashboard"
    page_icon: str = "⛰️"
    layout: str = "wide"
    
    # Map settings
    default_map_center: List[float] = None
    default_zoom_level: int = 8
    map_style: str = "OpenStreetMap"
    
    # Refresh settings
    auto_refresh_interval_seconds: int = 30
    max_data_points_plot: int = 1000
    
    # Color scheme
    risk_colors: Dict[str, str] = None
    
    def __post_init__(self):
        if self.default_map_center is None:
            self.default_map_center = [-25.2744, 133.7751]  # Australia center
        
        if self.risk_colors is None:
            self.risk_colors = {
                "CRITICAL": "#d62728",
                "HIGH": "#ff7f0e",
                "MEDIUM": "#ffbb78", 
                "LOW": "#2ca02c",
                "MINIMAL": "#98df8a"
            }

@dataclass
class MonitoringConfig:
    """System monitoring configuration"""
    health_check_interval_seconds: int = 60
    log_level: str = "INFO"
    log_file: str = "rockfall_system.log"
    max_log_size_mb: int = 100
    max_log_files: int = 5
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    slow_query_threshold_seconds: float = 1.0
    memory_usage_alert_threshold_percent: int = 85

@dataclass
class DataIngestionConfig:
    """Data ingestion and processing configuration"""
    # Sensor data validation ranges
    sensor_ranges: Dict[str, Dict[str, float]] = None
    
    # Data processing
    outlier_detection_enabled: bool = True
    outlier_z_score_threshold: float = 3.0
    missing_data_interpolation_method: str = "linear"
    
    # Batch processing
    batch_size: int = 1000
    processing_timeout_seconds: int = 300
    
    def __post_init__(self):
        if self.sensor_ranges is None:
            self.sensor_ranges = {
                "slope_angle": {"min": 0.0, "max": 90.0},
                "rock_strength": {"min": 0.0, "max": 100.0},
                "joint_spacing": {"min": 0.0, "max": 200.0},
                "joint_orientation": {"min": 0.0, "max": 360.0},
                "water_content": {"min": 0.0, "max": 100.0},
                "temperature": {"min": -50.0, "max": 60.0},
                "humidity": {"min": 0.0, "max": 100.0},
                "wind_speed": {"min": 0.0, "max": 100.0},
                "rainfall": {"min": 0.0, "max": 500.0},
                "vibration_intensity": {"min": 0.0, "max": 10.0},
                "latitude": {"min": -90.0, "max": 90.0},
                "longitude": {"min": -180.0, "max": 180.0}
            }

@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_api_key_auth: bool = False
    api_key: str = os.getenv("ROCKFALL_API_KEY", "")
    
    # Rate limiting
    enable_rate_limiting: bool = True
    requests_per_minute: int = 100
    
    # Data encryption
    encrypt_sensitive_data: bool = True
    encryption_key: str = os.getenv("ENCRYPTION_KEY", "")

class Config:
    """Main configuration class that combines all config sections"""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.model = ModelConfig()
        self.alerts = AlertConfig()
        self.api = APIConfig() 
        self.dashboard = DashboardConfig()
        self.monitoring = MonitoringConfig()
        self.data_ingestion = DataIngestionConfig()
        self.security = SecurityConfig()
        
        # Environment-specific overrides
        self._load_environment_overrides()
    
    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables"""
        
        # Database overrides
        if os.getenv("DB_PATH"):
            self.database.db_path = os.getenv("DB_PATH")
        
        # API overrides
        if os.getenv("API_HOST"):
            self.api.host = os.getenv("API_HOST")
        if os.getenv("API_PORT"):
            self.api.port = int(os.getenv("API_PORT"))
        if os.getenv("DEBUG"):
            self.api.debug = os.getenv("DEBUG").lower() == "true"
        
        # Alert overrides
        if os.getenv("ALERT_EMAIL_RECIPIENTS"):
            self.alerts.email_recipients = os.getenv("ALERT_EMAIL_RECIPIENTS").split(",")
        if os.getenv("ALERT_THRESHOLD"):
            self.alerts.critical_threshold = float(os.getenv("ALERT_THRESHOLD"))
        
        # Model overrides
        if os.getenv("MODEL_PATH"):
            self.model.model_path = os.getenv("MODEL_PATH")
        
        # Monitoring overrides
        if os.getenv("LOG_LEVEL"):
            self.monitoring.log_level = os.getenv("LOG_LEVEL")
    
    def get_alert_recipients(self, risk_level: str) -> Dict[str, List[str]]:
        """Get appropriate recipients based on risk level"""
        if risk_level in ["CRITICAL", "HIGH"]:
            return {
                "email": self.alerts.email_recipients,
                "sms": self.alerts.sms_recipients
            }
        elif risk_level == "MEDIUM":
            return {
                "email": self.alerts.email_recipients,
                "sms": []  # No SMS for medium risk
            }
        else:
            return {"email": [], "sms": []}
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check required email settings if email alerts enabled
        if self.alerts.email_enabled:
            if not self.alerts.smtp_username:
                issues.append("SMTP username not configured")
            if not self.alerts.smtp_password:
                issues.append("SMTP password not configured")
            if not self.alerts.email_recipients:
                issues.append("No email recipients configured")
        
        # Check SMS settings if enabled
        if self.alerts.sms_enabled:
            if not self.alerts.twilio_account_sid:
                issues.append("Twilio account SID not configured")
            if not self.alerts.twilio_auth_token:
                issues.append("Twilio auth token not configured")
            if not self.alerts.sms_recipients:
                issues.append("No SMS recipients configured")
        
        # Check model file exists
        if not os.path.exists(self.model.model_path):
            issues.append(f"Model file not found: {self.model.model_path}")
        
        # Validate risk thresholds
        thresholds = list(self.model.risk_thresholds.values())
        if thresholds != sorted(thresholds):
            issues.append("Risk thresholds must be in ascending order")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (useful for API responses)"""
        return {
            "database": self.database.__dict__,
            "model": self.model.__dict__,
            "alerts": {k: v for k, v in self.alerts.__dict__.items() 
                      if k not in ["smtp_password", "twilio_auth_token"]},  # Exclude sensitive data
            "api": self.api.__dict__,
            "dashboard": self.dashboard.__dict__,
            "monitoring": self.monitoring.__dict__,
            "data_ingestion": self.data_ingestion.__dict__,
            "security": {k: v for k, v in self.security.__dict__.items() 
                        if k not in ["api_key", "encryption_key"]}  # Exclude sensitive data
        }

# Global configuration instance
config = Config()

# Configuration validation function
def validate_startup_config():
    """Validate configuration at startup"""
    issues = config.validate_config()
    if issues:
        print("Configuration Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease check your configuration and environment variables.")
        return False
    return True

# Environment configuration presets
DEVELOPMENT_CONFIG = {
    "api.debug": True,
    "api.host": "localhost",
    "monitoring.log_level": "DEBUG",
    "alerts.email_enabled": False,
    "alerts.sms_enabled": False
}

PRODUCTION_CONFIG = {
    "api.debug": False,
    "api.host": "0.0.0.0",
    "monitoring.log_level": "INFO",
    "alerts.email_enabled": True,
    "security.enable_rate_limiting": True,
    "security.enable_api_key_auth": True
}

def apply_config_preset(preset: str):
    """Apply a configuration preset"""
    if preset.lower() == "development":
        preset_config = DEVELOPMENT_CONFIG
    elif preset.lower() == "production":
        preset_config = PRODUCTION_CONFIG
    else:
        raise ValueError(f"Unknown preset: {preset}")
    
    for key, value in preset_config.items():
        # Navigate nested configuration
        parts = key.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

if __name__ == "__main__":
    # Configuration validation script
    print("Rockfall Prediction System - Configuration Validation")
    print("=" * 60)
    
    if validate_startup_config():
        print("✅ Configuration is valid!")
    else:
        print("❌ Configuration has issues. Please fix before starting the system.")
    
    print(f"\nCurrent configuration summary:")
    print(f"- Database: {config.database.db_path}")
    print(f"- Model: {config.model.model_path}")
    print(f"- API: {config.api.host}:{config.api.port}")
    print(f"- Alerts: Email={config.alerts.email_enabled}, SMS={config.alerts.sms_enabled}")
    print(f"- Log Level: {config.monitoring.log_level}")
    print(f"- Debug Mode: {config.api.debug}")
