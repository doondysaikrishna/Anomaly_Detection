"""
AuraVision AI - Master Production Runner
Starts both the Unified Backend Server (Port 8000) and Frontend Application (Port 5173).
"""

import os
import sys
import time
import subprocess
import webbrowser
import signal
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).parent.resolve()
FRONTEND_DIR = ROOT_DIR / "frontend"
VENV_PYTHON = ROOT_DIR / ".venv" / "Scripts" / "python.exe"

PYTHON_EXE = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

def print_banner():
    print("=" * 70)
    print("  🚀 AURAVISION.AI - INDUSTRIAL VISUAL ANOMALY DETECTION PLATFORM")
    print("=" * 70)
    print("  📍 Frontend Web UI : http://localhost:5173")
    print("  🔌 Backend REST API: http://localhost:8000")
    print("  🧪 API Health Check: http://localhost:8000/api/health")
    print("=" * 70)
    print("  Press Ctrl+C at any time to gracefully stop all services.\n")

def main():
    print_banner()

    processes = []

    try:
        # 1. Start Backend Server
        print("[*] Launching Backend Server on http://localhost:8000...")
        backend_proc = subprocess.Popen(
            [PYTHON_EXE, "-u", "backend_server.py"],
            cwd=str(ROOT_DIR),
        )
        processes.append(("Backend", backend_proc))
        time.sleep(1.5)

        # 2. Start Frontend Dev Server
        print("[*] Launching Frontend Development Server on http://localhost:5173...")
        npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
        frontend_proc = subprocess.Popen(
            [npm_cmd, "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"],
            cwd=str(FRONTEND_DIR),
            shell=(os.name == "nt"),
        )
        processes.append(("Frontend", frontend_proc))
        time.sleep(2.0)

        print("\n[+] All services are active and running!")
        print("[*] Opening browser at http://localhost:5173 ...\n")
        try:
            webbrowser.open("http://localhost:5173")
        except Exception:
            pass

        # Keep parent script alive and monitor child processes
        while True:
            for name, proc in processes:
                poll = proc.poll()
                if poll is not None:
                    print(f"[!] {name} process exited with return code {poll}.")
                    return
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n[!] Shutdown signal received. Terminating all services...")
    finally:
        for name, proc in processes:
            try:
                print(f"[*] Stopping {name}...")
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        print("[+] All services stopped cleanly. Goodbye!\n")

if __name__ == "__main__":
    main()
