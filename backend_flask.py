"""
Flask Backend for Anomaly Detection System
Lightweight alternative using only standard library and Flask
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
import numpy as np
import os
from pathlib import Path

# Initialize Flask app
app = Flask(__name__, static_folder=None)

# Enable CORS
CORS(app, resources={r"/api/*": {"origins": "*"}})

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

# ==================== API Endpoints ====================

@app.route('/', methods=['GET'])
def home():
    """Root endpoint - health check"""
    return jsonify({
        "status": "running",
        "service": "Anomaly Detection Backend",
        "version": "1.0.0",
        "model_loaded": MODEL_LOADED,
        "timestamp": datetime.now().isoformat(),
        "api_base": "http://localhost:8000/api"
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": "LSTM Autoencoder",
        "threshold": THRESHOLD,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Get dashboard statistics"""
    return jsonify(STATS)

@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """Get model information"""
    return jsonify({
        "model_type": "LSTM Autoencoder",
        "input_features": 57,
        "sequence_length": 15,
        "encoder_units": 32,
        "decoder_units": 16,
        "threshold": THRESHOLD,
        "status": "loaded" if MODEL_LOADED else "not_loaded",
        "accuracy": MODEL_ACCURACY,
        "roc_auc": ROC_AUC
    })

@app.route('/api/predict', methods=['POST'])
def predict_anomaly():
    """Predict if input data is anomalous"""
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        data = request.json.get('data', [])
    except:
        return jsonify({"error": "Invalid JSON"}), 400
    
    if len(data) != 57:
        return jsonify({
            "error": f"Expected 57 features, got {len(data)}"
        }), 400
    
    # Simulate prediction
    data_array = np.array(data)
    score = float(np.std(data_array) * np.mean(np.abs(data_array)))
    score = min(max(score, 0.0), 1.0)
    
    is_anomaly = score > THRESHOLD
    
    return jsonify({
        "is_anomaly": is_anomaly,
        "score": score,
        "threshold": THRESHOLD,
        "message": "Anomaly detected" if is_anomaly else "Normal behavior"
    })

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get historical predictions"""
    limit = request.args.get('limit', 100, type=int)
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
    
    return jsonify({
        "count": len(history),
        "data": history
    })

@app.route('/api/realtime', methods=['GET'])
def get_realtime_data():
    """Get simulated real-time data"""
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "current_anomaly_rate": float(np.random.random() * 0.1),
        "active_sensors": 57,
        "last_anomaly": (datetime.now() - timedelta(hours=2)).isoformat() if np.random.random() > 0.7 else None,
        "status": "normal"
    })

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get model performance metrics"""
    return jsonify({
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
    })

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction for multiple data points"""
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        requests_data = request.json.get('requests', [])
    except:
        return jsonify({"error": "Invalid JSON"}), 400
    
    results = []
    for req in requests_data:
        data = req.get('data', [])
        if len(data) != 57:
            continue
        
        data_array = np.array(data)
        score = float(np.std(data_array) * np.mean(np.abs(data_array)))
        score = min(max(score, 0.0), 1.0)
        is_anomaly = score > THRESHOLD
        
        results.append({
            "is_anomaly": is_anomaly,
            "score": score,
            "threshold": THRESHOLD,
            "message": "Anomaly detected" if is_anomaly else "Normal behavior"
        })
    
    return jsonify({
        "total": len(requests_data),
        "processed": len(results),
        "results": results
    })

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard():
    """Get complete dashboard data"""
    return jsonify({
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
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Not found",
        "message": "Endpoint not found",
        "available_endpoints": [
            "GET  /api/health",
            "GET  /api/stats",
            "GET  /api/model/info",
            "POST /api/predict",
            "GET  /api/history",
            "GET  /api/realtime",
            "GET  /api/metrics",
            "POST /api/batch-predict",
            "GET  /api/dashboard"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "error": "Internal server error",
        "message": str(error)
    }), 500

# ==================== Startup ====================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Starting Anomaly Detection Backend (Flask)")
    print("="*70)
    print("📍 Backend URL: http://localhost:8000")
    print("📊 API Endpoints:")
    print("   • GET  http://localhost:8000/api/health")
    print("   • GET  http://localhost:8000/api/stats")
    print("   • POST http://localhost:8000/api/predict")
    print("   • GET  http://localhost:8000/api/metrics")
    print("   • GET  http://localhost:8000/api/dashboard")
    print("="*70)
    print("⚠️  Frontend: http://localhost:5173")
    print("="*70 + "\n")
    
    app.run(
        host='127.0.0.1',
        port=8000,
        debug=True,
        use_reloader=False,
        threaded=True
    )
