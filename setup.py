#!/usr/bin/env python3
"""
Setup script for Rockfall Prediction System
This script automates the installation and initial setup process
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import sqlite3
import json
from datetime import datetime

class RockfallSystemSetup:
    """Setup manager for the Rockfall Prediction System"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.venv_name = "rockfall_env"
        self.required_files = [
            "main.py",
            "app.py", 
            "train_model.py",
            "config.py",
            "requirements.txt"
        ]
    
    def print_banner(self):
        """Print setup banner"""
        print("=" * 60)
        print("🏔️  ROCKFALL PREDICTION SYSTEM SETUP")
        print("   AI-Powered Mining Safety Solution")
        print("=" * 60)
        print()
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        print("📋 Checking Python version...")
        
        if sys.version_info < (3, 8):
            print("❌ Error: Python 3.8 or higher is required")
            print(f"   Current version: {sys.version}")
            return False
        
        print(f"✅ Python {sys.version.split()[0]} detected")
        return True
    
    def check_required_files(self):
        """Check if all required files are present"""
        print("\n📁 Checking required files...")
        
        missing_files = []
        for file in self.required_files:
            if not (self.project_root / file).exists():
                missing_files.append(file)
                print(f"❌ Missing: {file}")
            else:
                print(f"✅ Found: {file}")
        
        if missing_files:
            print(f"\n❌ Missing {len(missing_files)} required files")
            print("Please ensure all files are in the project directory")
            return False
        
        return True
    
    def create_virtual_environment(self):
        """Create and activate virtual environment"""
        print(f"\n🔧 Setting up virtual environment '{self.venv_name}'...")
        
        venv_path = self.project_root / self.venv_name
        
        if venv_path.exists():
            print(f"⚠️  Virtual environment '{self.venv_name}' already exists")
            response = input("Do you want to recreate it? (y/N): ").strip().lower()
            if response == 'y':
                print("🗑️  Removing existing environment...")
                shutil.rmtree(venv_path)
            else:
                print("✅ Using existing virtual environment")
                return True
        
        try:
            # Create virtual environment
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
            print(f"✅ Virtual environment created: {venv_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False
    
    def install_requirements(self):
        """Install Python packages from requirements.txt"""
        print("\n📦 Installing Python packages...")
        
        venv_path = self.project_root / self.venv_name
        
        # Determine pip path based on OS
        if os.name == 'nt':  # Windows
            pip_path = venv_path / "Scripts" / "pip"
        else:  # Unix-like
            pip_path = venv_path / "bin" / "pip"
        
        try:
            # Upgrade pip first
            print("⬆️  Upgrading pip...")
            subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
            
            # Install requirements
            print("📥 Installing packages from requirements.txt...")
            subprocess.run([
                str(pip_path), "install", "-r", "requirements.txt"
            ], check=True)
            
            print("✅ All packages installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install packages: {e}")
            return False
    
    def setup_database(self):
        """Initialize the SQLite database"""
        print("\n🗄️  Setting up database...")
        
        db_path = self.project_root / "rockfall_data.db"
        
        try:
            conn = sqlite3.connect(str(db_path))
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
                    longitude REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
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
                    recommendations TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    site_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    risk_level TEXT,
                    probability REAL,
                    alert_sent BOOLEAN DEFAULT FALSE,
                    alert_method TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sensor_site_time ON sensor_readings(site_id, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pred_site_time ON predictions(site_id, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_site_time ON alerts(site_id, timestamp)')
            
            conn.commit()
            conn.close()
            
            print("✅ Database initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Database setup failed: {e}")
            return False
    
    def create_directories(self):
        """Create necessary directories"""
        print("\n📂 Creating project directories...")
        
        directories = [
            "data",
            "models", 
            "logs",
            "uploads",
            "exports",
            "backups"
        ]
        
        for dir_name in directories:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(exist_ok=True)
            print(f"✅ Created: {dir_name}/")
        
        return True
    
    def create_env_file(self):
        """Create .env file with default configuration"""
        print("\n⚙️  Creating environment configuration...")
        
        env_path = self.project_root / ".env"
        
        if env_path.exists():
            print("⚠️  .env file already exists")
            response = input("Do you want to overwrite it? (y/N): ").strip().lower()
            if response != 'y':
                print("✅ Keeping existing .env file")
                return True
        
        env_content = """# Rockfall Prediction System Configuration
# Copy this file and customize for your environment

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False

# Database Configuration  
DB_PATH=rockfall_data.db

# Model Configuration
MODEL_PATH=rockfall_model.joblib

# Email Alert Configuration
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL_RECIPIENTS=admin@yourcompany.com,safety@yourcompany.com

# SMS Alert Configuration (Twilio)
TWILIO_ACCOUNT_SID=your-twilio-sid
TWILIO_AUTH_TOKEN=your-twilio-token
TWILIO_PHONE_NUMBER=+1234567890

# Security Configuration
ROCKFALL_API_KEY=your-secret-api-key
ENCRYPTION_KEY=your-encryption-key

# Alert Thresholds
ALERT_THRESHOLD=0.7

# Logging
LOG_LEVEL=INFO
"""
        
        try:
            with open(env_path, 'w') as f:
                f.write(env_content)
            print("✅ .env file created")
            print("⚠️  Please edit .env file with your actual configuration values")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    
    def generate_sample_data(self):
        """Generate sample training data"""
        print("\n🎲 Generating sample data and training model...")
        
        try:
            # Import and run training script
            venv_path = self.project_root / self.venv_name
            
            # Determine python path based on OS
            if os.name == 'nt':  # Windows
                python_path = venv_path / "Scripts" / "python"
            else:  # Unix-like
                python_path = venv_path / "bin" / "python"
            
            # Run training script to generate data and model
            print("🤖 Running ML training pipeline...")
            result = subprocess.run([
                str(python_path), "train_model.py", "demo"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Sample data and model generated")
                return True
            else:
                print(f"⚠️  Training script output: {result.stdout}")
                print(f"❌ Training failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to generate sample data: {e}")
            return False
    
    def create_startup_scripts(self):
        """Create convenience startup scripts"""
        print("\n📜 Creating startup scripts...")
        
        # Windows batch script
        batch_content = """@echo off
echo Starting Rockfall Prediction System...
echo.

echo Starting FastAPI backend...
start "FastAPI Backend" cmd /k "cd /d %~dp0 && %~dp0\\rockfall_env\\Scripts\\python.exe main.py"

echo Waiting for backend to start...
timeout /t 5 /nobreak > nul

echo Starting Streamlit dashboard...
start "Streamlit Dashboard" cmd /k "cd /d %~dp0 && %~dp0\\rockfall_env\\Scripts\\streamlit.exe run app.py"

echo.
echo System started! 
echo - FastAPI backend: http://localhost:8000
echo - Streamlit dashboard: http://localhost:8501
echo.
pause
"""
        
        # Unix shell script
        shell_content = """#!/bin/bash
echo "Starting Rockfall Prediction System..."
echo

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo "Port $1 is already in use"
        return 1
    fi
    return 0
}

# Check ports
check_port 8000 || exit 1
check_port 8501 || exit 1

echo "Starting FastAPI backend..."
source ./rockfall_env/bin/activate
python main.py &
FASTAPI_PID=$!

echo "Waiting for backend to start..."
sleep 5

echo "Starting Streamlit dashboard..."
streamlit run app.py &
STREAMLIT_PID=$!

echo
echo "System started!"
echo "- FastAPI backend: http://localhost:8000"
echo "- Streamlit dashboard: http://localhost:8501"
echo
echo "Press Ctrl+C to stop both services"

# Wait for interrupt
trap "kill $FASTAPI_PID $STREAMLIT_PID; exit" INT
wait
"""
        
        try:
            # Create Windows batch file
            with open("start_system.bat", 'w') as f:
                f.write(batch_content)
            
            # Create Unix shell script
            with open("start_system.sh", 'w') as f:
                f.write(shell_content)
            
            # Make shell script executable
            if os.name != 'nt':
                os.chmod("start_system.sh", 0o755)
            
            print("✅ Startup scripts created:")
            print("   - start_system.bat (Windows)")
            print("   - start_system.sh (Linux/Mac)")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create startup scripts: {e}")
            return False
    
    def create_readme(self):
        """Create README file with usage instructions"""
        print("\n📄 Creating README file...")
        
        readme_content = """# Rockfall Prediction System

AI-powered rockfall prediction and monitoring system for open-pit mines.

## Quick Start

### Option 1: Use Startup Scripts
- **Windows**: Double-click `start_system.bat`
- **Linux/Mac**: Run `./start_system.sh`

### Option 2: Manual Start
```bash
# Activate virtual environment
source rockfall_env/bin/activate  # Linux/Mac
# OR
rockfall_env\\Scripts\\activate    # Windows

# Start FastAPI backend (Terminal 1)
python main.py

# Start Streamlit dashboard (Terminal 2)
streamlit run app.py
```

## Access Points

- **API Documentation**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501
- **Health Check**: http://localhost:8000/health

## Configuration

Edit `.env` file to configure:
- Email alerts (SMTP settings)
- SMS alerts (Twilio settings)
- Alert thresholds
- Database paths

## Usage

### 1. Upload Sensor Data
- Use the dashboard "Data Upload" page
- Upload CSV files with sensor readings
- Or use manual data entry form

### 2. View Predictions
- Check "Overview" for system-wide status
- Use "Site Analysis" for detailed site monitoring
- Monitor real-time risk levels and alerts

### 3. Configure Alerts
- Set up email/SMS notifications
- Configure risk thresholds
- Test alert delivery

## Data Format

CSV files should include these columns:
- site_id, slope_angle, rock_strength, joint_spacing
- joint_orientation, water_content, temperature
- humidity, wind_speed, rainfall, vibration_intensity
- latitude, longitude

## API Endpoints

- `POST /predict` - Get rockfall prediction
- `GET /predictions/{site_id}` - Get site history
- `POST /upload/sensors` - Upload sensor data
- `POST /alerts/config` - Configure alerts

## Troubleshooting

1. **Port already in use**: Change ports in config.py
2. **Model not found**: Run `python train_model.py` first
3. **Database errors**: Check file permissions
4. **Email alerts not working**: Verify SMTP settings in .env

## Support

For issues and feature requests, check the project documentation.
"""
        
        try:
            with open("README.md", 'w') as f:
                f.write(readme_content)
            print("✅ README.md created")
            return True
        except Exception as e:
            print(f"❌ Failed to create README: {e}")
            return False
    
    def run_setup(self):
        """Run the complete setup process"""
        self.print_banner()
        
        steps = [
            ("Check Python version", self.check_python_version),
            ("Check required files", self.check_required_files),
            ("Create virtual environment", self.create_virtual_environment),
            ("Install requirements", self.install_requirements),
            ("Setup database", self.setup_database),
            ("Create directories", self.create_directories),
            ("Create .env file", self.create_env_file),
            ("Generate sample data", self.generate_sample_data),
            ("Create startup scripts", self.create_startup_scripts),
            ("Create README", self.create_readme)
        ]
        
        success_count = 0
        for step_name, step_func in steps:
            if step_func():
                success_count += 1
            else:
                print(f"\n❌ Setup failed at step: {step_name}")
                break
        
        print(f"\n{'='*60}")
        if success_count == len(steps):
            print("🎉 SETUP COMPLETED SUCCESSFULLY!")
            print("\nNext steps:")
            print("1. Edit .env file with your configuration")
            print("2. Run startup script or start services manually")
            print("3. Access dashboard at http://localhost:8501")
            print("\nFor detailed instructions, see README.md")
        else:
            print(f"⚠️  Setup partially completed ({success_count}/{len(steps)} steps)")
            print("Please resolve the issues and run setup again")
        
        print(f"{'='*60}")

def main():
    """Main setup function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Rockfall Prediction System Setup")
        print("Usage: python setup.py")
        print("\nThis script will:")
        print("- Create virtual environment")
        print("- Install required packages")
        print("- Initialize database")
        print("- Generate sample data")
        print("- Create configuration files")
        return
    
    setup = RockfallSystemSetup()
    setup.run_setup()

if __name__ == "__main__":
    main()
