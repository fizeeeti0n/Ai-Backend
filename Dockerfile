# Use an official Python runtime as a parent image
# We are upgrading from 'buster' to 'bullseye' because buster repositories are EOL.
# python:3.9-slim-bullseye provides a stable and supported base.
FROM python:3.9-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Install Tesseract OCR and its language data, plus essential image processing libraries
# libjpeg-dev for JPEG support, zlib1g-dev for PNG support (used by Pillow)
RUN apt-get update && \
    apt-get install -y tesseract-ocr libtesseract-dev libjpeg-dev zlib1g-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the working directory
COPY requirements.txt .

# Install any needed Python packages specified in requirements.txt
# --no-cache-dir reduces image size by not storing build artifacts
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the working directory
COPY . .

# Set the Tesseract command path as an environment variable for pytesseract
# This path is where tesseract-ocr executable is typically installed on Debian-based systems
ENV TESSERACT_CMD_PATH=/usr/bin/tesseract

# Expose the port your Flask app runs on (Gunicorn's default is 8000)
EXPOSE 8000

# Define environment variables for Flask and Gunicorn
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Command to run the Flask application using Gunicorn
# 'app:app' means it will look for a Flask app instance named 'app' in 'app.py'
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
