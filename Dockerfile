# Use an official Python runtime as a parent image
# We choose a version compatible with Render's typical Python versions and your app
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install Tesseract OCR and its language data
# This step requires root privileges, which are available during Docker image build
# The 'libtesseract-dev' package provides development headers, which might be needed
# by pytesseract's underlying C components, though 'tesseract-ocr' provides the executable.
RUN apt-get update && \
    apt-get install -y tesseract-ocr libtesseract-dev && \
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
