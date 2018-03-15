
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

from PIL import Image, ImageTk
import requests
from io import BytesIO
from tkinter import Tk,Label,Canvas,NW,Entry,Button 
url = ''
window = Tk()
window.title("Welcome to Image predictor") 
window.geometry('800x600')
lbl = Label(window, text="Enter the URL of the image", font=("Helvetica", 16))
lbl.pack()
def clicked(): 
    global url
    lbl.configure()
    url  = (User_input.get())
    print(url)
    response = requests.get(url)
    test_image = Image.open(BytesIO(response.content))
    put_image = test_image.resize((400,400)) 
    test_image = test_image.resize((128,128))  
    img = ImageTk.PhotoImage(put_image)
    pic = Label(image=img)
    pic.pack()
    pic.image = img
    test_image = image_utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    
 
    result = model.predict_on_batch(test_image)

    if result[0][0] == 1:
        ans = 'french fries'
    elif result[0][1] == 1:
        ans = 'pizza'
    elif result[0][2] == 1:
        ans = 'samosa'
    out = Label(window, text  = 'Predicted answer : ' +  ans, font=("Helvetica", 16))
    out.pack()

User_input = Entry(width = 100)
User_input.pack()
btn = Button(window, text="Detect Image", font=("Helvetica", 12), command=clicked)
btn.pack()
window.mainloop()


