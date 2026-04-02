import threading
import webbrowser
import time
from app import app

def open_browser():
    # Wait a second for Flask to boot up before opening the page
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    print("🚀 Booting up the Stress System Web App...")
    # Start the browser thread
    threading.Thread(target=open_browser, daemon=True).start()
    # Run the Flask app
    app.run(host='127.0.0.1', port=5000, debug=False)