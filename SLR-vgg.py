from google.colab import drive
drive.mount('/content/drive')
!unzip /content/drive/MyDrive/SLR/archive.zip
# Importing the Keras libraries and packages
from keras.applications.vgg19 import VGG19
from keras.application.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input, Lambda ,Dense ,Flatten ,Dropout
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
import os
import cv2
train_dir = "/content/training_set" #file-path-of-training-dataset.
eval_dir = "/content/test_set"
#Helper function to load images from given directories
def load_images(directory):
    images_test = [] #testing data
    labels = []   # initialize two empty lists to store images and labels. /training data
    for idx, label in enumerate(uniq_labels):
        for file in os.listdir(directory + "/" + label):
            filepath = directory + "/" + label + "/" + file #building the path for file
            image = cv2.resize(cv2.imread(filepath), (64, 64)) #reads the image using opencv and resizes it to 64x64 pixels.
            images_test.append(image)
            labels.append(idx)
    images_test = np.array(images_test)   #convert the lists to NumPy arrays.
    labels = np.array(labels)
    return(images_test, labels)    #returns the NumPy arrays containing images and labels.
uniq_labels = sorted(os.listdir(train_dir))
images, labels = load_images(directory = train_dir)
if uniq_labels == sorted(os.listdir(eval_dir)):
    X_eval, y_eval = load_images(directory = eval_dir)
#Training and Testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, stratify = labels)

n = len(uniq_labels)
train_n = len(X_train)
test_n = len(X_test)

print("Total number of signs ", n)
print("Number of training images: " , train_n)
print("Number of testing images: ", test_n)

eval_n = len(X_eval)
print("Number of evaluation images: ", eval_n)
#one hot encoding in 2D array
y_train = keras.utils.to_categorical(y_train) #it converts the class vector to binary matrix.
y_test = keras.utils.to_categorical(y_test)
y_eval = keras.utils.to_categorical(y_eval)
X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0
X_eval = X_eval.astype('float32')/255.0
classifier_vgg19= VGG19(input_shape= (64,64,3),include_top=False,weights='imagenet')
for layer in classifier_vgg19.layers:
  layer.trainable= False
#VGG19/VGG16
classifier= classifier_vgg19.output
classifier = Flatten()(classifier)#elongating the pooling map
classifier= Dropout (0.4)(classifier)
classifier= Dense (, activation="relu")(classifier)
classifier = Dense(unit=36, activation='softmax')(classifier)
model= model(inputs= classifier_vgg19.input, outputs= classifier)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary() #VGG19
history1 = model.fit(X_train, y_train, epochs =4, batch_size = 64,validation_data=(X_test,y_test))
#VGG19
score = model.evaluate(x = X_test, y = y_test, verbose = 0)
print('Accuracy for test images:', round(score[1]*100, 3), '%')
score = model.evaluate(x = X_eval, y = y_eval, verbose = 0)
print('Accuracy for evaluation images:', round(score[1]*100, 3), '%')
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('model accuracy of VGG19')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model loss of VGG19')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

!wget -q -O - ipv4.icanhazip.com
! streamlit run main.py & npx localtunnel --port 8501
