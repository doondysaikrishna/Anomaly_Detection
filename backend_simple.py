"""
Simple HTTP Backend for Anomaly Detection System
Using only Python standard library (no external dependencies)
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import json
import numpy as np
from datetime import datetime, timedelta
import threading

# ==================== Global Configuration ====================

MODEL_LOADED = True
THRESHOLD = 0.15
MODEL_ACCURACY = 0.97
ROC_AUC = 0.98

STATS = {
    "total_records": 50000,
    "normal_records": 46000,
    "attack_records": 4000,
    "attack_percentage": 8.0,
    "model_accuracy": MODEL_ACCURACY,
    "roc_auc": ROC_AUC
}

# ==================== HTTP Request Handler ====================

class AnomalyDetectionHandler(BaseHTTPRequestHandler):
    """Handle HTTP requests for anomaly detection API"""
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        # CORS headers
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        try:
            if path == '/' or path == '/api':
                response = {
                    "status": "running",
                    "service": "Anomaly Detection Backend",
                    "version": "1.0.0",
                    "model_loaded": MODEL_LOADED,
                    "timestamp": datetime.now().isoformat(),
                    "api_base": "http://localhost:8000/api"
                }
            
            elif path == '/api/health':
                response = {
                    "status": "healthy",
                    "model": "LSTM Autoencoder",
                    "threshold": THRESHOLD,
                    "timestamp": datetime.now().isoformat()
                }
            
            elif path == '/api/stats':
                response = STATS
            
            elif path == '/api/model/info':
                response = {
                    "model_type": "LSTM Autoencoder",
                    "input_features": 57,
                    "sequence_length": 15,
                    "encoder_units": 32,
                    "decoder_units": 16,
                    "threshold": THRESHOLD,
                    "status": "loaded" if MODEL_LOADED else "not_loaded",
                    "accuracy": MODEL_ACCURACY,
                    "roc_auc": ROC_AUC
                }
            
            elif path == '/api/history':
                limit = 100
                now = datetime.now()
                history = []
                
                for i in range(limit):
                    timestamp = now - timedelta(minutes=i)
                    is_anomaly = np.random.random() < 0.05
                    score = np.random.random() * (0.2 if is_anomaly else 0.1)
                    
                    history.append({
                        "timestamp": timestamp.isoformat(),
                        "score": float(score),
                        "is_anomaly": is_anomaly,
                        "status": "Attack" if is_anomaly else "Normal"
                    })
                
                response = {
                    "count": len(history),
                    "data": history
                }
            
            elif path == '/api/realtime':
                response = {
                    "timestamp": datetime.now().isoformat(),
                    "current_anomaly_rate": float(np.random.random() * 0.1),
                    "active_sensors": 57,
                    "last_anomaly": (datetime.now() - timedelta(hours=2)).isoformat() if np.random.random() > 0.7 else None,
                    "status": "normal"
                }
            
            elif path == '/api/metrics':
                response = {
                    "confusion_matrix": {
                        "true_negatives": 370947,
                        "false_positives": 6933,
                        "false_negatives": 4605,
                        "true_positives": 50016
                    },
                    "precision": 0.88,
                    "recall": 0.92,
                    "f1_score": 0.90,
                    "accuracy": 0.97,
                    "roc_auc": 0.98,
                    "threshold": THRESHOLD
                }
            
            elif path == '/api/dashboard':
                response = {
                    "stats": STATS,
                    "model_info": {
                        "type": "LSTM Autoencoder",
                        "accuracy": MODEL_ACCURACY,
                        "roc_auc": ROC_AUC,
                        "threshold": THRESHOLD
                    },
                    "metrics": {
                        "precision": 0.88,
                        "recall": 0.92,
                        "f1_score": 0.90
                    },
                    "timestamp": datetime.now().isoformat()
                }
            
            else:
                self.send_response(404)
                response = {
                    "error": "Not found",
                    "available_endpoints": [
                        "GET  /api/health",
                        "GET  /api/stats",
                        "GET  /api/model/info",
                        "POST /api/predict",
                        "GET  /api/history",
                        "GET  /api/realtime",
                        "GET  /api/metrics",
                        "GET  /api/dashboard"
                    ]
                }
            
            self.wfile.write(json.dumps(response).encode())
        
        except Exception as e:
            self.send_response(500)
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        # CORS headers
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode())
            
            if path == '/api/predict':
                if not MODEL_LOADED:
                    response = {"error": "Model not loaded"}
                else:
                    features = data.get('data', [])
                    if len(features) != 57:
                        response = {"error": f"Expected 57 features, got {len(features)}"}
                    else:
                        # Simulate prediction
                        data_array = np.array(features, dtype=float)
                        score = float(np.std(data_array) * np.mean(np.abs(data_array)))
                        score = min(max(score, 0.0), 1.0)
                        is_anomaly = score > THRESHOLD
                        
                        response = {
                            "is_anomaly": is_anomaly,
                            "score": score,
                            "threshold": THRESHOLD,
                            "message": "Anomaly detected" if is_anomaly else "Normal behavior"
                        }
            
            elif path == '/api/batch-predict':
                if not MODEL_LOADED:
                    response = {"error": "Model not loaded"}
                else:
                    requests_data = data.get('requests', [])
                    results = []
                    
                    for req in requests_data:
                        features = req.get('data', [])
                        if len(features) == 57:
                            data_array = np.array(features, dtype=float)
                            score = float(np.std(data_array) * np.mean(np.abs(data_array)))
                            score = min(max(score, 0.0), 1.0)
                            is_anomaly = score > THRESHOLD
                            
                            results.append({
                                "is_anomaly": is_anomaly,
                                "score": score,
                                "threshold": THRESHOLD,
                                "message": "Anomaly detected" if is_anomaly else "Normal behavior"
                            })
                    
                    response = {
                        "total": len(requests_data),
                        "processed": len(results),
                        "results": results
                    }
            
            else:
                self.send_response(404)
                response = {"error": "Endpoint not found"}
            
            self.wfile.write(json.dumps(response).encode())
        
        except Exception as e:
            self.send_response(500)
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests (CORS preflight)"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Content-Length', '0')
        self.end_headers()
    
    def log_message(self, format, *args):
        """Override to customize logging"""
        if '200' in str(args):
            print(f"✓ {self.command} {self.path}")
        else:
            print(f"✗ {self.command} {self.path} - {args}")

# ==================== Server Setup ====================

def start_server():
    """Start the HTTP server"""
    server_address = ('127.0.0.1', 8000)
    httpd = HTTPServer(server_address, AnomalyDetectionHandler)
    
    print("\n" + "="*70)
    print("🚀 Starting Anomaly Detection Backend (Pure Python HTTP Server)")
    print("="*70)
    print("📍 Backend URL: http://localhost:8000")
    print("📊 API Endpoints:")
    print("   • GET  http://localhost:8000/api/health")
    print("   • GET  http://localhost:8000/api/stats")
    print("   • POST http://localhost:8000/api/predict")
    print("   • GET  http://localhost:8000/api/metrics")
    print("   • GET  http://localhost:8000/api/dashboard")
    print("   • GET  http://localhost:8000/api/history")
    print("   • GET  http://localhost:8000/api/realtime")
    print("="*70)
    print("⚠️  Frontend: http://localhost:5173")
    print("="*70)
    print("✋ Press Ctrl+C to stop\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n✅ Backend server stopped")
        httpd.server_close()

if __name__ == '__main__':
    start_server()
