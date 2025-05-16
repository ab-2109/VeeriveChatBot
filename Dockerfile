# Use Python 3.12 base image
FROM python:3.12.3-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=/app

WORKDIR /app

# Install necessary system dependencies and Chrome for Selenium
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    supervisor \
    wget \
    curl \
    gnupg \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcups2 \
    libgdk-pixbuf2.0-0 \
    libnspr4 \
    libnss3 \
    libxcomposite1 \
    libxdamage1 \
    xdg-utils \
    libxrandr2 \
    libgbm1 \
    libxshmfence1 \
    libdrm2 \
    libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/*

# Install Google Chrome for Selenium
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google.list && \
    apt-get update && \
    apt-get install -y google-chrome-stable && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Add a non-root user for better security
RUN useradd -m appuser
RUN mkdir -p /app/data /app/logs && chown -R appuser:appuser /app

# Copy the rest of your application code
COPY . .

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Switch to non-root user
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Add healthcheck for container
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start both FastAPI and qdrant_maintainer using supervisord
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
