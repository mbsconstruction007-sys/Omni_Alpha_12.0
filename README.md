# 🚀 Omni Alpha 5.0 - Advanced 24-Step Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

A comprehensive FastAPI-based web application for managing and executing a 24-step analysis process with bot integration, webhook capabilities, and advanced analytics.

## 🚀 Features

- **24-Step Analysis Process**: Complete workflow management for complex analysis tasks
- **RESTful API**: FastAPI-based backend with comprehensive endpoints
- **Bot Integration**: Webhook support for bot interactions
- **Real-time Advice**: Dynamic recommendations based on analysis progress
- **Testing Suite**: Comprehensive test scripts for all components

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## 🛠️ Installation

1. **Clone or download the project**
   ```bash
   cd "omni alpha"
   ```

2. **Run the setup script**
   ```bash
   python setup.py
   ```

3. **Activate the virtual environment**
   
   **Windows (PowerShell):**
   ```powershell
   .venv\Scripts\activate
   ```
   
   **Linux/MacOS:**
   ```bash
   source .venv/bin/activate
   ```

## 🏃‍♂️ Running the Application

### Terminal A - Start the API Server
```powershell
# Activate virtual environment
.venv\Scripts\activate

# Start the FastAPI server
uvicorn src.app:app --host 127.0.0.1 --port 8000

# For development with auto-reload
uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
```

### Terminal B - Run Tests
```powershell
# Activate virtual environment
.venv\Scripts\activate

# Set environment variable
$env:BOT_BASE_URL = "http://127.0.0.1:8000"

# Run test scripts
python check_step4_endpoints.py
python check_step7_webhook.py
python check_step8_advice.py
```

## 📡 API Endpoints

### Core Endpoints
- `GET /` - Root endpoint with API information
- `GET /health` - Health check endpoint
- `GET /steps` - Get all 24 analysis steps
- `GET /steps/{step_id}` - Get specific analysis step
- `POST /analysis/start` - Start a new analysis
- `POST /steps/{step_id}/complete` - Mark a step as completed

### Bot Integration
- `POST /webhook` - Webhook endpoint for bot integration

### Advice & Recommendations
- `GET /advice` - Get analysis advice and recommendations

## 🧪 Testing

The project includes comprehensive test scripts:

### Step 4 - Endpoints Testing
```bash
python check_step4_endpoints.py
```
Tests all core API endpoints and functionality.

### Step 7 - Webhook Testing
```bash
python check_step7_webhook.py
```
Tests webhook functionality and bot integration.

### Step 8 - Advice Testing
```bash
python check_step8_advice.py
```
Tests advice and recommendation system.

## 📁 Project Structure

```
omni alpha/
├── src/
│   ├── __init__.py
│   └── app.py              # Main FastAPI application
├── check_step4_endpoints.py    # Endpoint testing script
├── check_step7_webhook.py      # Webhook testing script
├── check_step8_advice.py       # Advice testing script
├── requirements.txt            # Python dependencies
├── setup.py                   # Setup script
└── README.md                  # This file
```

## 🔧 Configuration

The application uses environment variables for configuration:

- `BOT_BASE_URL`: Base URL for bot integration (default: http://127.0.0.1:8000)

## 📊 API Documentation

Once the server is running, you can access:
- **Interactive API docs**: http://127.0.0.1:8000/docs
- **ReDoc documentation**: http://127.0.0.1:8000/redoc

## 🚀 Development Workflow

1. **Start the API server** in Terminal A
2. **Run tests** in Terminal B to verify functionality
3. **Make changes** to the code
4. **Test changes** using the test scripts
5. **Iterate** as needed

## 📝 Notes

- The application uses in-memory storage for demo purposes
- All test scripts include comprehensive error handling
- The 24-step analysis process is fully configurable
- Webhook endpoints support various event types for bot integration

## 🤝 Contributing

1. Make changes to the code
2. Run the test scripts to ensure functionality
3. Update documentation as needed

## 📄 License

This project is part of the Omni Alpha analysis system.
