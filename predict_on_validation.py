import cv2
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
 
CATEGORIES =['paper','rock', 'scissor']

# Loading the model
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

def prepare(filepath):
    IMG_SIZE = 150
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)/255.0
    return new_array
    
    

for i in range(1, 34):
    prediction = loaded_model.predict([prepare('validation/{}.png'.format(i))])
    print(i)
    print(CATEGORIES[np.argmax(prediction[0])])
  

