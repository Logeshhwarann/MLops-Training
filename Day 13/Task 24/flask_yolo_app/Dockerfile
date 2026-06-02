FROM python:3.11-slim

WORKDIR /app

# Install system deps for OpenCV / Ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create directories
RUN mkdir -p static/uploads static/results

EXPOSE 5000

CMD ["python", "app.py"]
