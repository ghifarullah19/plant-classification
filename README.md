# Plant Classification

This project is a Flask-based backend application that leverages Deep Learning to classify plant images into 29 different species. The model is built using Transfer Learning with the **VGG16** architecture and is containerized using Docker for easy deployment.

## Resources
  - [Plat Classification Model Development](https://github.com/ghifarullah19/proyek-massive-botanify/blob/main/Notebook/plant-classification.ipynb)

## Features
  - **Image Classification:** Identifies plant species from uploaded images.
  - **Top-5 Predictions:** Returns the top 5 most likely classes with their probability scores.
  - **REST API:** Provides simple endpoints for integration with frontend or mobile applications.
  - **Dockerized:** Ready for deployment using Docker containers.

## Tech Stack
  - **Language:** Python 3.10
  - **Web Framework:** Flask
  - **Machine Learning:** TensorFlow 2.18.0, Keras (VGG16)
  - **Image Processing:** OpenCV, Numpy
  
## Supported Plant Classes
The model is trained to recognize the following 29 plant species:
Aster, Euphorbia, Bergamot, Sage, Azalea, Peony, Pansy, Orchid, Dandelion, Cosmos, Snapdragon, Polyanthus, Dahlia, Gerbera, Ixora, Eustoma, Daisy, Aglaonema, Sunflower, Viola, Lily Flower, Rose, Iris, Tuberose, Alyssum, Dieffenbacia, Jasmine, Lavender, Tulip.

## Installation and Local Setup
### Prerequisites
  - Python 3.10
  - Git LFS (Large File Storage) is required to download the model weights.

### Steps
1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <project-directory>
    ```

2.  **Download Model Weights:**
    This project uses Git LFS for the model file (`myvgg16_model.h5`). Ensure you pull the actual large file:
    ```bash
    git lfs pull
    ```

3.  **Set up a Virtual Environment (Optional):**
    ```bash
    python -m venv venv
    # Linux/Mac
    source venv/bin/activate
    # Windows
    venv\Scripts\activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Application:**
    ```bash
    python app.py
    ```
    The application will start in debug mode at `http://127.0.0.1:5000`.

## API Documentation
### 1\. Health Check
Verifies that the server is running.
  - **URL:** `/`
  - **Method:** `GET`
  - **Response:**
    ```json
    {
      "message": "Hello, World!",
      "success": 1
    }
    ```

### 2\. Detect Plant
Classifies an uploaded image.
  - **URL:** `/detect`
  - **Method:** `POST`
  - **Body (form-data):**
      - `image`: The image file to classify (supported formats: jpg, png, etc.).
  - **Success Response (200 OK):**
    ```json
    {
      "label": "Rose",
      "probability": 0.985,
      "result": [
          {"label": "Rose", "probability": 0.985},
          {"label": "Tulip", "probability": 0.010},
          ...
      ],
      "message": "Image classified successfully",
      "success": 1
    }
    ```
  - **Error Response:**
      - If no image is provided: `{"error": 1, "message": "Image is required"}`

## Project Structure
```text
.
├── app.py                # Main Flask application and API endpoints
├── pipeline.py           # Script for model creation and training
├── Dockerfile            # Instructions to build the Docker image
├── Procfile              # Deployment command for platforms like Heroku
├── requirements.txt      # List of Python dependencies
├── runtime.txt           # Python version specification
├── class_names.pkl       # Serialized list of class labels
├── myvgg16_model.h5      # Pre-trained Deep Learning model (stored via Git LFS)
└── .gitattributes        # Git LFS configuration
```

## Model Details
The model is defined in `pipeline.py` with the following specifications:
  - **Base Architecture:** VGG16 (pre-trained on ImageNet).
  - **Input Shape:** (224, 224, 3).
  - **Modifications:** The base VGG16 layers are frozen to retain learned features.
  - **Custom Head:**
      - Flatten layer
      - Dense layer (256 units, ReLU activation)
      - Output Dense layer (29 units, Softmax activation)
  - **Training:** The model is compiled with the Adam optimizer and sparse categorical crossentropy loss.
