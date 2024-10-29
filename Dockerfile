# Use a lightweight Python image as the base
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the necessary files to the container
COPY requirements.txt requirements.txt
COPY app.py app.py
COPY pipeline.py pipeline.py
COPY class_names.pkl class_names.pkl
COPY botanify_model_vgg16_v3.keras botanify_model_vgg16_v3.keras

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the app runs on
EXPOSE 8080

# Command to run the app with Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "4"]
