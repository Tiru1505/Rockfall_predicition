#!/usr/bin/env python3
"""
Quick Start Script for Rockfall Prediction System
Run this to automatically start both backend and dashboard
"""

import subprocess
import sys
import time
import webbrowser
import os
from pathlib import Path

def check_files():
    """Check if all required files exist"""
    required_files = ['main.py', 'app.py', 'requirements.txt']
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all files are in the current directory.")
        return False
    
    print("✅ All required files found!")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def start_backend():
    """Start FastAPI backend"""
    print("🚀 Starting FastAPI backend...")
    try:
        # Use the fixed main.py file
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return process
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None

def start_dashboard():
    """Start Streamlit dashboard"""
    print("📊 Starting Streamlit dashboard...")
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return process
    except Exception as e:
        print(f"❌ Failed to start dashboard: {e}")
        return None

def main():
    print("🏔️  Rockfall Prediction System - Quick Start")
    print("=" * 50)
    
    # Check files
    if not check_files():
        return
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed. Please install requirements manually:")
        print("   pip install -r requirements.txt")
        return
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        return
    
    print("⏳ Waiting for backend to start...")
    time.sleep(8)
    
    # Start dashboard
    dashboard_process = start_dashboard()
    if not dashboard_process:
        backend_process.terminate()
        return
    
    print("⏳ Waiting for dashboard to start...")
    time.sleep(5)
    
    print("\n🎉 System started successfully!")
    print("-" * 50)
    print("📊 Dashboard: http://localhost:8501")
    print("🔧 API Docs:  http://localhost:8000/docs")
    print("💓 Health:    http://localhost:8000/health")
    print("-" * 50)
    print("Press Ctrl+C to stop both services")
    
    # Open browser
    try:
        webbrowser.open("http://localhost:8501")
    except:
        pass
    
    # Wait for processes
    try:
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("❌ Backend process stopped")
                break
            if dashboard_process.poll() is not None:
                print("❌ Dashboard process stopped")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        backend_process.terminate()
        dashboard_process.terminate()
        
        # Wait for processes to terminate
        backend_process.wait(timeout=5)
        dashboard_process.wait(timeout=5)
        
        print("✅ Services stopped successfully!")

if __name__ == "__main__":
    main()