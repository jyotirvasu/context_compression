# ─────────────────────────────────────────────────────────
# Context Compression Pipeline — Docker Image
# ─────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="jsharm30"
LABEL description="Context Compression Showcase Pipeline"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .

# Install Python dependencies + Flask
RUN pip install --no-cache-dir -r requirements.txt flask gunicorn && \
    python -m nltk.downloader punkt punkt_tab

# Copy application code
COPY . .

# Generate showcase HTML at build time (avoids runtime model issues)
RUN python showcase.py || true

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120", "app:app"]
