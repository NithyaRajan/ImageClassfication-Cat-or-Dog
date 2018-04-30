
from keras import applications
from keras import optimizer

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense

from keras.preprocessing.image import ImageDataGenerator

import numpy as nump
#--------------------------------------------------------

#This is method #2 which uses the bottleneck features of the VGG16 pretrained model

#Our primary goal with this method is to reach a better accuracy in a short amount of time
#For our pre-trained network, we will use VGG16 architecture, which is a pre-trained model on an ImageNet dataset


#Defining a few things: constants for image dimensions, epochs, batch size, and the number of training/test images we have
width, height = 150, 150
epochs = 50
batch_size = 16
num_train = 2000
num_test = 800

#Here, we are defining the paths to the directories where we have our training/test data as well as our model
model_path = '/Users/nithyarajan/Cat_or_Dog/bottleneck_fc_model.h5'
train_data_path = '/Users/nithyarajan/Downloads/dataset/training_set'
test_data_path = '/Users/nithyarajan/Downloads/dataset/testing_set'

#Function to generate data and bottleneck features for the VGG16 Model
def bottleneck_features():
    #The ImageDataGenerator provided by Keras generates minibatches of image data to use for preparation and augmentation
    imgdata = ImageDataGenerator(rescale=1. / 255)

    #Invoking and building the VGG16 Model
    VGG16model = applications.VGG16(include_top=False, weights='imagenet')

    #Here, we are using flow_from_directory to generate the augmented/normalized image data
    #The data that is generated here will infinitely loop in batches indefinitely
    gen = imgdata.flow_from_directory(train_data_path, target_size = (width, height), batch_size = batch_size, class_mode=None, shuffle=False)

    #The predict_generator function generates predictions for the input samples from a data generator (which we used above)
    bottleneck_features_train = model.predict_generator(gen, num_train // batch_size)
    nump.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)

    #Doing the same thing for the test data
    gen = imgdata.flow_from_directory(test_data_path, target_size = (width, height), batch_size = batch_size, class_mode=None, shuffle=False)
    bottleneck_features_test = model.predict_generator(gen, num_test // batch_size)
    nump.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_test)


#Function to train the VGG16 Model
def train_top_model():

    #Loading the training data from a numpy array into a variable
    train_data = nump.load(open('bottleneck_features_train.npy'))
    train_labels = nump.array([0] * (num_train / 2) + [1] * (num_train / 2))

    #Doing the same thing for the test data
    test_data = nump.load(open('bottleneck_features_validation.npy'))
    test_labels = nump.array([0] * (num_test / 2) + [1] * (num_test / 2))

    #We are now creating a Sequential model through the Sequential constructor, and passing in the layers
    model = Sequential([
        Flatten(input_shape=train_data.shape[1:]),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
        ])

    #After adding everything to the model, the compile function configures the model for training
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    #Now the fit function will train the model for the number of epochs we have set (50)
    model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(test_data, test_labels))
    model.save_weights(model_path)


#Function calls to execute/train the model
bottleneck_features()
train_top_model()