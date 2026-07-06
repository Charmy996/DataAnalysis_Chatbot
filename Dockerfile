FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create necessary folders
RUN mkdir -p reports output chunks images

# HF Spaces runs as non-root user 1000
RUN chown -R 1000:1000 /app
USER 1000

EXPOSE 7860

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]
