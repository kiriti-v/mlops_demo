# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Download the model
RUN mkdir -p ../1-model/saved_model
COPY ../1-model/download_model.py ../1-model/
RUN cd ../1-model && python download_model.py

# Set environment variables
ENV PORT=8080
ENV MODEL_PATH=/app/1-model/saved_model

# Expose the port
EXPOSE 8080

# Run the application
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app 