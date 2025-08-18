# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create storage directory
RUN mkdir -p storage

# Expose ports
EXPOSE 8000 8501

# Create a startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Function to start FastAPI\n\
start_fastapi() {\n\
    echo "Starting FastAPI server..."\n\
    python app.py &\n\
    FASTAPI_PID=$!\n\
    echo "FastAPI started with PID: $FASTAPI_PID"\n\
}\n\
\n\
# Function to start Streamlit\n\
start_streamlit() {\n\
    echo "Starting Streamlit server..."\n\
    streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true &\n\
    STREAMLIT_PID=$!\n\
    echo "Streamlit started with PID: $STREAMLIT_PID"\n\
}\n\
\n\
# Handle termination\n\
cleanup() {\n\
    echo "Shutting down services..."\n\
    kill $FASTAPI_PID $STREAMLIT_PID 2>/dev/null\n\
    wait\n\
    exit 0\n\
}\n\
\n\
trap cleanup SIGTERM SIGINT\n\
\n\
# Start services based on mode\n\
if [ "$1" = "api" ]; then\n\
    echo "Starting in API mode"\n\
    exec python app.py\n\
elif [ "$1" = "streamlit" ]; then\n\
    echo "Starting in Streamlit mode"\n\
    exec streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true\n\
elif [ "$1" = "pipeline" ]; then\n\
    echo "Running pipeline directly"\n\
    exec python pipeline.py\n\
else\n\
    echo "Starting both FastAPI and Streamlit"\n\
    start_fastapi\n\
    start_streamlit\n\
    \n\
    # Wait for any process to exit\n\
    wait -n\n\
    \n\
    # Exit with status of process that exited first\n\
    exit $?\n\
fi' > /app/start.sh

# Make startup script executable
RUN chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["/app/start.sh"]