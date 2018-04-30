from keras.models import Sequential
from keras.models import load_model

from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

import numpy as nump
#--------------------------------------------------------

#This is method #1 which will train our model on our data using a convnet from the baseline

#Defining a few things: epochs and batch size
epochs = 2
steps_per_epoch = 4000
batch_size = 32

#Definitions of paths for the model as well as the training/test data
model_path = '/Users/nithyarajan/Cat_or_Dog/my_model.h5'
train_data_path = '/Users/nithyarajan/Downloads/dataset/training_set'
test_data_path = '/Users/nithyarajan/Downloads/dataset/testing_set'

#Building a Sequential CNN model, and passing in the layers
model = Sequential([
    Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu')
    MaxPooling2D(pool_size = (2, 2))
    Conv2D(32, (3, 3), activation = 'relu')
    MaxPooling2D(pool_size = (2, 2))
    Flatten()
    Dense(units = 128, activation = 'relu')
    Dense(units = 1, activation = 'sigmoid')
    ])

#After adding everything to the model, the compile function configures the model for training
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#The ImageDataGenerator provided by Keras generates minibatches of image data to use for preparation and augmentation
train_imgdata = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
training_set = train_imgdata.flow_from_directory(train_data_path, target_size = (64, 64), batch_size = batch_size, class_mode = 'binary')

#Generating for test data as well
test_imgdata = ImageDataGenerator(rescale = 1./255)
test_set = test_imgdata.flow_from_directory(test_data_path, target_size = (64, 64), batch_size = batch_size, class_mode = 'binary')

#Fit the model using the fit_generator method
model.fit_generator(training_set, steps_per_epoch = steps_per_epoch, epochs = epochs, validation_data = test_set, validation_steps = 226)

model.save(model_path)
del model

#Load the model
model = load_model(model_path)

#Loading an image to classify/use for predictions
test_image = image.load_img('/Users/nithyarajan/Downloads/dataset/single_prediction/cat_or_dog_3.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = nump.expand_dims(test_image, axis = 0)

#This will make a prediction on the given test image, which we will confirm
result = model.predict(test_image)
training_set.class_indices

#Check the image to confirm if its a dog or cat, then print the result to show that the classification was correct
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)

#Print a summary representation of the model
model.summary()

model.output_shape
