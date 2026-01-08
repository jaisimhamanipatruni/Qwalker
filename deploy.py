# deploy.py - One-click deployment script
import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ Requirements installed successfully!")

def run_application():
    """Run the Dash application"""
    print("🚀 Starting EPANET Analyzer...")
    print("🌐 Open your browser and go to: http://localhost:8050")
    print("📊 The application is now running!")
    subprocess.call([sys.executable, "app.py"])

def main():
    """Main deployment function"""
    print("=" * 60)
    print("🌊 EPANET Anomaly & Quantum Walk Analyzer")
    print("=" * 60)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        create_requirements_file()
    
    # Install requirements
    try:
        install_requirements()
    except Exception as e:
        print(f"❌ Error installing requirements: {e}")
        return
    
    # Run application
    try:
        run_application()
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

def create_requirements_file():
    """Create requirements.txt if it doesn't exist"""
    requirements = """dash==2.11.0
dash-bootstrap-components==1.4.1
plotly==5.14.1
numpy==1.24.0
pandas==2.0.0
networkx==3.0
scipy==1.10.0
flask==2.3.0"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("📝 Created requirements.txt file")

if __name__ == "__main__":
    main()
