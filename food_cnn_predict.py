
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

#load model
img_width, img_height = 128, 128
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

#Prediction on a new picture
from keras.preprocessing import image as image_utils

from PIL import Image
import requests
from io import BytesIO

response = requests.get('https://food.fnr.sndimg.com/content/dam/images/food/fullset/2016/5/11/0/FNM_060116-Double-Fried-French-Fries_s4x3.jpg.rend.hgtvcom.616.462.suffix/1463001459282.jpeg')
test_image = Image.open(BytesIO(response.content))
test_image = test_image.resize((128,128))
#test_image = image_utils.load_img('dataset/test2.jpg', target_size=(128, 128))
test_image = image_utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
 
result = model.predict_on_batch(test_image)
if result[0][0] == 1:
    print('french fries')
elif result[0][1] == 1:
    print('pizza')
elif result[0][2] == 1:
    print('samosa')