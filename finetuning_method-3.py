from keras import applications
from keras import optimizers

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense

from keras.preprocessing.image import ImageDataGenerator
#--------------------------------------------------------

#This is method #3 where we are fine-tuning the top layers of a pre-trained network.

#What this means is that we will be improving our accuracy by following a few steps:
#Step 1: Instantiate the convolution base of VGG16 and load its weights
#Step 2: Add our defined, fully connected model on top and load its weights
#Step 3: Freeze the layers of the VGG16 model up to the last convolutional block


#Defining a few things: constants for image dimensions, epochs, batch size, and the number of training/test images we have
width, height = 150, 150
num_train = 2000
num_test = 800
epochs = 50
batch_size = 16

#Here, we are defining the paths to the directories where we have our training/test data as well as our model and weights
weights_path = '/Users/nithyarajan/Cat_or_Dog/vgg16_weights.h5'
model_path = '/Users/nithyarajan/Cat_or_Dog/fc_model.h5'
train_data_path = '/Users/nithyarajan/Downloads/dataset/training_set'
test_data_path = '/Users/nithyarajan/Downloads/dataset/testing_set'


#Invoking and building the VGG16 Model
model = applications.VGG16(weights='imagenet', include_top=False)
print('Model loaded.')


#Building a Sequential model that goes on top of the VGG16 Model
top_layer = Sequential([
    Flatten(input_shape=model.output_shape[1:])
    Dense(256, activation='relu')
    Dropout(0.5)
    Dense(1, activation='sigmoid')
    ])


#Loading the weights on the top layer because we need a fully-trained classifier to fine-tune
top_layer.load_weights(model_path)

#Add the Sequential model top layer we instantiated to our VGG16 Model
model.add(top_layer)

#A simple for loop in which we set the first 25 layers as non-trainable
for layer in model.layers[:25]:
    layer.trainable = False

#Compile the model with an SGD optimizer
model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])

#The ImageDataGenerator provided by Keras generates minibatches of image data to use for preparation and augmentation
train_imgdata = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

#Do the same ImageDataGenerator generating for the test data
test_imgdata = ImageDataGenerator(rescale=1. / 255)

#Here, we are using flow_from_directory to generate the augmented/normalized image data for the training data
train_gen = train_imgdata.flow_from_directory(train_data_path, target_size=(height, width), batch_size=batch_size, class_mode='binary')

#Use the same flow_from_directory function to generate the augmented/normalized image data for the test data
test_gen = test_imgdata.flow_from_directory(test_data_path, target_size=(height, width), batch_size=batch_size, class_mode='binary')

#We will now finally fine-tune the model using the fit_generator function
model.fit_generator(train_gen, samples_per_epoch=num_train, epochs=epochs, validation_data=test_gen, nb_val_samples=num_test)