from flask import Flask, render_template, Response, jsonify, request
from logic import StressEngine

app = Flask(__name__)
engine = StressEngine()
engine.start()

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    while True:
        frame = engine.latest_frame_jpeg
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/data')
def get_data():
    return jsonify(engine.get_state())

@app.route('/api/toggle_mic', methods=['POST'])
def toggle_mic():
    data = request.json
    engine.use_laptop_mic = data.get('use_laptop_mic', False)
    src = "LAPTOP" if engine.use_laptop_mic else "ESP32"
    engine.log_event(f"Audio Source Switched to: {src}")
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=False, port=5000)