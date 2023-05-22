from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image

app = Flask(__name__)

# load the pre-trained model
with open('./cnn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# define a function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET'])
def index():
    # Render the index.html page as the landing page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    prediction = None
    image_path = None
    if 'file' in request.files:
        # save the uploaded file
        file = request.files['file']
        image_path = 'static/images/uploaded_image.jpg'
        file.save(image_path)
        # preprocess the image
        image = preprocess_image(image_path)
        # make a prediction using the model
        prediction = model.predict(image)[0]
    return render_template('flask.html', prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
