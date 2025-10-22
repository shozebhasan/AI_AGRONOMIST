
"""
Development server runner for Agronomist AI API
"""

import uvicorn
from api import app

if __name__ == "__main__":
    print("🚀 Starting Agronomist AI API Server...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("⏹️  Press Ctrl+C to stop the server\n")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )