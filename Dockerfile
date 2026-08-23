# Lightweight Python 3.11 Slim Base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and scripts
COPY src/ /app/src/
COPY data/ /app/data/

# Create data directory for volume persistence
RUN mkdir -p /app/data

# Default port for dashboard
EXPOSE 5000

# Default command (can be overridden by docker-compose)
CMD ["python", "src/main.py", "--daemon"]
