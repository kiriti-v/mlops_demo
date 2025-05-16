# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create model directory
RUN mkdir -p model/saved_model

# Copy the model download script and the application
COPY download_model.py model/
COPY app.py .

# Download the model
RUN cd model && python download_model.py

# Set environment variables
ENV PORT=8080
ENV MODEL_PATH=/app/model/saved_model

# Expose the port
EXPOSE 8080

# Run the application
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app 