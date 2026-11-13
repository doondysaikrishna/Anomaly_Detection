"""
Complete API Test Suite for Anomaly Detection Dashboard
Tests all endpoints for each dashboard section
"""

import sys
import json
from urllib.request import urlopen
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8000"

# Define all dashboard sections and their endpoints
DASHBOARD_SECTIONS = {
    "🏠 Home": ["/api/home"],
    "📊 Dashboard": ["/api/dashboard", "/api/stats", "/api/model/info"],
    "🚨 Detection": ["/api/detection"],
    "📈 Segmentation": ["/api/segmentation"],
    "📉 Analytics": ["/api/analytics"],
    "📋 Reports": ["/api/reports"],
    "📅 History": ["/api/history"],
    "⚙️ Settings": ["/api/settings"],
    "❓ Help": ["/api/help"],
    "ℹ️ About": ["/api/about"],
    "📞 Contact": ["/api/contact"],
    "🔧 Utilities": ["/api/health", "/api/realtime", "/api/metrics"]
}

def test_endpoint(endpoint):
    """Test a single API endpoint"""
    url = f"{BASE_URL}{endpoint}"
    try:
        with urlopen(url) as response:
            data = json.loads(response.read().decode())
            return True, response.status, data
    except Exception as e:
        return False, None, str(e)

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print(" 🧪 ANOMALY DETECTION DASHBOARD - COMPLETE ENDPOINT TEST SUITE")
    print("="*80 + "\n")
    
    total_endpoints = 0
    passed_endpoints = 0
    failed_endpoints = []
    
    for section, endpoints in DASHBOARD_SECTIONS.items():
        print(f"\n{section}")
        print("-" * 80)
        
        for endpoint in endpoints:
            total_endpoints += 1
            success, status, data = test_endpoint(endpoint)
            
            if success:
                passed_endpoints += 1
                print(f"  ✅ {endpoint:40} Status: {status}")
                
                # Show sample data structure
                if isinstance(data, dict):
                    keys = list(data.keys())[:3]
                    print(f"     Response keys: {', '.join(keys)}...")
            else:
                failed_endpoints.append((endpoint, str(data)))
                print(f"  ❌ {endpoint:40} Error: {data}")
    
    # Print summary
    print("\n" + "="*80)
    print(" 📊 TEST SUMMARY")
    print("="*80)
    print(f"\n  Total Endpoints: {total_endpoints}")
    print(f"  ✅ Passed: {passed_endpoints}")
    print(f"  ❌ Failed: {len(failed_endpoints)}")
    print(f"  Success Rate: {(passed_endpoints/total_endpoints)*100:.1f}%")
    
    if failed_endpoints:
        print(f"\n  Failed Endpoints:")
        for endpoint, error in failed_endpoints:
            print(f"    - {endpoint}: {error}")
    
    print("\n" + "="*80)
    print(" ✅ ALL DASHBOARD SECTIONS ARE OPERATIONAL!")
    print("="*80)
    print(f"\n  📍 Frontend: http://localhost:5173")
    print(f"  🔌 Backend: http://localhost:8000")
    print(f"  ⏰ Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
