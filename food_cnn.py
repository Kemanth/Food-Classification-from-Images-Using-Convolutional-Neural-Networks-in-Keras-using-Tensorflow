from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
 
#Initialize the CNN
classifier = Sequential()
#Convolution and Max pooling
classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
 
#Flatten
classifier.add(Flatten())
 
#Full connection
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(3, activation = 'softmax'))
 
#Compile classifier
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
 
#Fitting CNN to the images
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True) 
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('./dataset/training_set', target_size=(128, 128), batch_size=32, class_mode='categorical')
test_set = test_datagen.flow_from_directory('./dataset/test_set', target_size=(128, 128), batch_size=32, class_mode='categorical')
classifier.fit_generator(training_set, steps_per_epoch=800/32, epochs=50, validation_data=test_set, validation_steps = 200/32)

#save model
import os
target_dir = './models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
classifier.save('./models/model.h5')
classifier.save_weights('./models/weights.h5')
