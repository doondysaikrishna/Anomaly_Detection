#!/usr/bin/env python3
import http.server
import socketserver
import os
from pathlib import Path

# Change to the frontend dist directory
os.chdir(Path(__file__).parent / 'frontend' / 'dist')

PORT = 5173
Handler = http.server.SimpleHTTPRequestHandler

print(f"🚀 Frontend Server Starting...")
print(f"📍 Access the frontend at: http://localhost:{PORT}")
print(f"📁 Serving from: {os.getcwd()}")
print(f"⏹️  Press Ctrl+C to stop the server\n")

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n✋ Server stopped")
