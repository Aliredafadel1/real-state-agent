# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Set environment variables
# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies (minimal for lean image)
# Add packages here if needed, e.g.: build-essential gcc

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files (all directories except those in .dockerignore)
COPY . .

# Expose ports (adjust as needed)
# FastAPI: 8000
# Streamlit: 8501
EXPOSE 8000 8501

# Default command - can be overridden
CMD ["sh", "-c", "streamlit run ui/app.py --server.address 0.0.0.0 --server.port ${PORT}"]
