import os
import uuid
import json
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load YOLO model - tries best.pt first, falls back to yolov8n.pt"""
    try:
        from ultralytics import YOLO
        # Try user's custom model first
        if os.path.exists('best.pt'):
            print("✅ Loading custom model: best.pt")
            return YOLO('best.pt'), 'custom'
        else:
            print("⚠️  best.pt not found. Downloading yolov8n.pt (COCO 80-class model)...")
            return YOLO('yolov8n.pt'), 'coco'
    except ImportError:
        print("❌ Ultralytics not installed. Run: pip install ultralytics")
        return None, None

# Load model at startup
model, model_type = load_model()

@app.route('/')
def index():
    model_status = 'custom' if model_type == 'custom' else ('coco' if model_type == 'coco' else 'unavailable')
    return render_template('index.html', model_status=model_status)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please install ultralytics: pip install ultralytics'}), 500

    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'}), 400

    results_data = []

    for file in files:
        if not file or file.filename == '':
            continue
        if not allowed_file(file.filename):
            results_data.append({
                'filename': file.filename,
                'error': 'File type not supported'
            })
            continue

        try:
            # Save uploaded file
            uid = str(uuid.uuid4())[:8]
            ext = file.filename.rsplit('.', 1)[1].lower()
            save_name = f"{uid}.{ext}"
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
            file.save(upload_path)

            # Run YOLO prediction
            conf_threshold = float(request.form.get('confidence', 0.25))
            results = model(upload_path, conf=conf_threshold, verbose=False)

            # Save result image
            result_name = f"result_{uid}.jpg"
            result_path = os.path.join(app.config['RESULTS_FOLDER'], result_name)
            results[0].save(filename=result_path)

            # Extract detections
            detections = []
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls_id] if cls_id < len(model.names) else f"class_{cls_id}"
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                    detections.append({
                        'label': label,
                        'confidence': round(conf * 100, 1),
                        'bbox': [round(x1), round(y1), round(x2), round(y2)]
                    })

            # Sort by confidence descending
            detections.sort(key=lambda d: d['confidence'], reverse=True)

            results_data.append({
                'original_name': file.filename,
                'upload_url': f"/static/uploads/{save_name}",
                'result_url': f"/static/results/{result_name}",
                'detections': detections,
                'total_objects': len(detections),
                'model_type': model_type
            })

        except Exception as e:
            results_data.append({
                'filename': file.filename,
                'error': str(e)
            })

    return jsonify({'results': results_data})

@app.route('/model-info')
def model_info():
    if model is None:
        return jsonify({'status': 'unavailable'})
    return jsonify({
        'status': 'loaded',
        'type': model_type,
        'classes': len(model.names) if hasattr(model, 'names') else 0,
        'class_names': list(model.names.values()) if hasattr(model, 'names') else []
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
