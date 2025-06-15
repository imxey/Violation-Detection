from flask import Flask, render_template, request
import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model('model_multiclass.h5')
with open('labels.json', 'r') as f:
    label_map = json.load(f)
class_labels = list(label_map.keys())

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    max_pred = prediction[0][np.argmax(prediction)]
    return {
        "prediction": np.round(prediction, 3).tolist(),
        "max_prediction": float(max_pred),
        "class_label": class_labels[np.argmax(prediction)],
        "detected": max_pred > 0.5 and class_labels[np.argmax(prediction)] != 'bukan_pelanggaran'
    }

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    image_url = None
    fileName= None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            result = predict_image(filepath)
            image_url = filepath
            fileName = file.filename
    return render_template("index.html", result=result, image_url=image_url, filepath=fileName,  class_labels=class_labels)

if __name__ == '__main__':
    app.run(debug=True)
