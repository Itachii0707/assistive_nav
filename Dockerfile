# Use a lightweight python image
FROM python:3.10-slim

# Install system dependencies needed for OpenCV, GL, etc.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install CPU-only PyTorch first to prevent huge GPU/CUDA downloads (saves gigabytes of bandwidth and RAM on the host)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 5000

# Set environment variables for production
ENV PORT=5000
ENV YOLO_AUTO_INIT=true

# Start gunicorn with gevent worker
CMD ["gunicorn", "-k", "gevent", "-w", "1", "-b", "0.0.0.0:5000", "web.app:app"]
