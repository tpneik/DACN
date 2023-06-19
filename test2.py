from keras.models import load_model
import urllib.request
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

import numpy as np

path = "model.h5"
model = load_model(path)

image_path = "tt"

image_name = os.listdir(image_path)

print(image_name)
print("------------------------------------------------------------------------------------------------------")
for imag in image_name:
    #file_path = os.path.join(image_path, imag)
    img_path = os.path.join(image_path, imag)
    print(img_path)
    
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    class_names = ['Tomato___Late_blight', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Apple___Apple_scab', 'Corn_(maize)___healthy', 'Soybean___healthy', 'Potato___healthy', 'Apple___Cedar_apple_rust', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Strawberry___Leaf_scorch', 'Corn_(maize)___Northern_Leaf_Blight', 'Cherry_(including_sour)___Powdery_mildew', 'Peach___Bacterial_spot', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Raspberry___healthy', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___healthy', 'Grape___Esca_(Black_Measles)', 'Orange___Haunglongbing_(Citrus_greening)', 'Potato___Late_blight', 'Strawberry___healthy', 'Pepper,_bell___Bacterial_spot', 'Apple___healthy', 'Peach___healthy', 'Tomato___Tomato_mosaic_virus', 'Squash___Powdery_mildew', 'Pepper,_bell___healthy', 'Corn_(maize)___Common_rust_', 'Tomato___Leaf_Mold', 'Cherry_(including_sour)___healthy', 'Grape___healthy', 'Blueberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Apple___Black_rot', 'Grape___Black_rot', 'Potato___Early_blight', 'Tomato___Target_Spot', 'Tomato___Septoria_leaf_spot']


    probs = model.predict(img)[0]

    pred_class_prob = np.argmax(probs)
    pred_class_name = class_names[pred_class_prob]

    print(f'Predicted class: {pred_class_name}')
    print(f'Probability: {probs[pred_class_prob]}')
    print("------------------------------------------------------------------------------------------------------")