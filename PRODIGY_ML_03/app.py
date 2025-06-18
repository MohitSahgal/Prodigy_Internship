from flask import Flask, request, render_template
import joblib
import cv2
import numpy as np
import os

app = Flask(__name__)
MODEL_PATH = 'svm_cat_dog_model.joblib'
IMAGE_SIZE = (128, 128)

# Loading our trained model
model = joblib.load(MODEL_PATH)


def preprocess_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0
    img_flatten = img.reshape(1, -1)
    return img_flatten


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', prediction="No file part")

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', prediction="No selected file")

        # Save the uploaded file temporarily
        filepath = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(filepath)

        # Save a copy for static preview
        static_preview_path = os.path.join('static', 'preview.jpg')
        cv2.imwrite(static_preview_path, cv2.imread(filepath))

        # Preprocess and predict
        img_data = preprocess_image(filepath)
        if img_data is None:
            prediction = "Invalid image"
        else:
            pred = model.predict(img_data)[0]
            prediction = "Dog" if pred == 1 else "Cat"

        # Optionally, remove the file after prediction
        os.remove(filepath)

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
