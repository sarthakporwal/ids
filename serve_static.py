
import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

PORT = 8080
DIRECTORY = Path(__file__).parent / "static"

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def main():
    os.chdir(DIRECTORY)
    
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print("=" * 60)
        print("🛡️  CANShield Static Web Interface")
        print("=" * 60)
        print(f"\n✅ Server started at: http://localhost:{PORT}")
        print(f"📁 Serving from: {DIRECTORY}")
        print("\n💡 Features:")
        print("   - Interactive dashboard")
        print("   - Drag & drop file upload")
        print("   - Real-time visualizations")
        print("   - Attack detection simulation")
        print("\n⚠️  Note: This is the static HTML version")
        print("   For full functionality, use:")
        print("   • Streamlit app: streamlit run app.py")
        print("   • REST API: python api_backend.py")
        print("\nPress Ctrl+C to stop the server\n")
        print("=" * 60)
        
        webbrowser.open(f'http://localhost:{PORT}')
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Server stopped")

if __name__ == "__main__":
    main()

