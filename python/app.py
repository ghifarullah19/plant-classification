from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import cv2
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        if request.files.get('image') is None or request.files['image'].filename == '':
            return jsonify(error=1, message='Image is required')
        # Load model
        MODEL_PATH = "myvgg16_model.h5"
        # Load pickle (ini adalah nama class atau label yang digunakan)
        PICKLE_PATH = "class_names.pkl"

        # Isi dari class_names.pkl: 
        # class_names = ['Aster', 'Euphorbia', 'Bergamot', 'Sage', 'Azalea', 'Peony', 'Pansy', 'Orchid', 'Dandelion', 'Cosmos', 'Snapdragon', 'Polyanthus', 'Dahlia', 'Gerbera', 'Ixora', 'Eustoma', 'Daisy', 'Aglaonema', 'Sunflower', 'Viola', 'Lily Flower', 'Rose', 'Iris', 'Tuberose', 'Alyssum', 'Dieffenbacia', 'Jasmine', 'Lavender', 'Tulip']

        filestr = request.files['image'].read()
        npimg = np.fromstring(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # pre-process the image for classification
        image = cv2.resize(image, (224, 224))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # load the trained CNN and the label binarizer
        print("[INFO] loading network...")
        model = load_model(MODEL_PATH)
        lb = pickle.loads(open(PICKLE_PATH, "rb").read())
        # lb = class_names

        # classify the input image
        print("[INFO] classifying image...")
        # return the 5 hightest probability class
        proba = model.predict(image)[0]
        idxs = np.argsort(proba)[::-1][:5]

        # loop over the indexes of the high confidence class labels
        result = []
        for (i, j) in enumerate(idxs):
            # build the result list
            result.append({"label": lb[j], "probability": float(proba[j])})

        label = lb[np.argmax(proba)]
        probability = float(proba[np.argmax(proba)])
        return jsonify(success=1, result=result, message='Image classified successfully', label=label, probability=probability)

    return jsonify(error=1, message='Unsupported HTTP method')

@app.route('/')
def index():
    return jsonify(success=1, message='Hello, World!')

if __name__ == '__main__':
    app.run(debug=True)