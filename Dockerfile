# Use a lightweight Python image as the base
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the necessary files to the container
COPY requirements.txt requirements.txt
COPY python/app.py app.py
COPY python/pipeline.py pipeline.py
COPY python/class_names.pkl class_names.pkl
COPY python/myvgg16_model.h5 myvgg16_model.h5

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the app runs on
EXPOSE 8080

# Command to run the app with Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "4"]
