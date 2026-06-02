# YOLO Vision — Flask Object Detection App

A polished Flask web app that runs YOLOv8 object detection on uploaded images.

---

## 📁 Project Structure

```
flask_yolo_app/
├── app.py                  # Flask backend
├── requirements.txt
├── Dockerfile
├── best.pt                 # ← Place your custom YOLO model here
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    ├── js/app.js
    ├── uploads/            # Auto-created
    └── results/            # Auto-created
```

---

## 🚀 Quick Start (Local)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your model (optional)
Place your `best.pt` file in the project root.  
If not present, the app **auto-downloads YOLOv8n** (COCO 80 classes) on first run.

### 3. Run
```bash
python app.py
```

Open → http://localhost:5000

---

## 🐳 Docker

```bash
# Build
docker build -t yolo-vision .

# Run (with your model)
docker run -p 5000:5000 -v $(pwd)/best.pt:/app/best.pt yolo-vision

# Run (without model — uses YOLOv8n)
docker run -p 5000:5000 yolo-vision
```

---

## ☁️ AWS EC2 Deployment

### 1. Launch EC2
- AMI: Ubuntu 22.04 LTS
- Instance: t3.medium (2 vCPU, 4GB RAM) minimum
- Security Group: Open port **5000** (or 80 with nginx proxy)

### 2. SSH & install Docker
```bash
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>

# Install Docker
sudo apt update && sudo apt install -y docker.io
sudo systemctl enable docker
sudo usermod -aG docker ubuntu
# Log out and back in
```

### 3. Copy files & run
```bash
# From your local machine
scp -i your-key.pem -r flask_yolo_app ubuntu@<EC2_PUBLIC_IP>:~/

# On EC2
cd ~/flask_yolo_app
docker build -t yolo-vision .
docker run -d -p 5000:5000 --name yolo-app yolo-vision
```

### 4. Access
```
http://<EC2_PUBLIC_IP>:5000
```

---

## 🔧 Features

- **Multi-image upload** — drag & drop or browse, select multiple files
- **Live preview** — thumbnails before detection
- **Confidence threshold slider** — 0.05 to 0.95
- **Side-by-side view** — original vs annotated result
- **Detection list** — all objects with confidence bars
- **Fullscreen zoom** — click any result image
- **Model fallback** — uses YOLOv8n if best.pt not found

---

## 🎯 Using Your Custom `best.pt`

The app loads `best.pt` from the project root automatically.  
Class names are read directly from your model's metadata — no config needed.

---

## API Endpoint

`POST /predict`  
- Form fields: `files` (multiple), `confidence` (float 0.05–0.95)  
- Returns JSON with detections, bounding boxes, confidence scores, result image URLs.
