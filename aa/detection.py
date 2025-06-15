import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

model = load_model('model_multiclass.h5')

with open('labels.json', 'r') as f:
    label_map = json.load(f)
class_labels = list(label_map.keys())

img = image.load_img('aa.jpeg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
prediction = model.predict(img_array)
max_pred = prediction[0][np.argmax(prediction)]

print("Prediction (softmax output):", np.round(prediction,3))
print("Hasil prediksi terbesar", max_pred)
if max_pred > 0.5 and class_labels[np.argmax(prediction)]!= 'bukan_pelanggaran':
  print("TERDETEKSI PELANGGARAN!!")
  print("Hasil prediksi:", class_labels[np.argmax(prediction)])
else:
  print("Tidak ada pelanggaran")
