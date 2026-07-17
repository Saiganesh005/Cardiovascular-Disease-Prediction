# ==========================================================
# Cardio-Predict
# Docker Configuration
# ==========================================================

FROM python:3.11-slim

LABEL maintainer="Robbi Sai Ganesh Devi Prasad"

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Display logs immediately
ENV PYTHONUNBUFFERED=1

# Working Directory
WORKDIR /app

# Install System Dependencies
RUN apt-get update && \
    apt-get install -y gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency file
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy complete project
COPY . .

# Expose Flask Port
EXPOSE 5000

# Flask Environment Variables
ENV FLASK_APP=app/app.py
ENV FLASK_ENV=production

# Start Application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app.app:app"]
