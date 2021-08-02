
import numpy as np       #Importing necessary librairies
import os
import matplotlib.pyplot as plt
import keras
import pandas as pd
import cv2
import seaborn as sns

images_path = 'drive/MyDrive/CK+48_dataset'        #Path of the dataset present in google drive
print('The different emotions to be detected are: ') 
print(os.listdir(images_path))

total = sum(len(files) for _, _, files in os.walk(r'drive/MyDrive/CK+48_dataset'))        #Getting the total count of the images present
print('Total images in dataset: ')
print(total)

total_images = []                                  #Loading all the images
pos = np.ones((total,),dtype='int64')
features = []
j = 0
path_anger = images_path + '/anger'          #Loading anger dataset
anger_img = os.listdir(path_anger)
features.append('anger')
for i in anger_img:
  img = cv2.imread(path_anger + '/' + i)
  img = cv2.resize(img,(48,48))
  total_images.append(img)
  pos[j] = 0
  j = j+1
print('Loaded images from anger dataset')

path_fear = images_path + '/fear'          #Loading fear dataset
fear_img = os.listdir(path_fear)
features.append('fear')
for i in fear_img:
  img = cv2.imread(path_fear + '/' + i)
  img = cv2.resize(img,(48,48))
  total_images.append(img)
  pos[j] = 1
  j = j+1
print('Loaded images from fear dataset')

path_happy = images_path + '/happy'          #Loading happy dataset
happy_img = os.listdir(path_happy)
features.append('happy')
for i in happy_img:
  img = cv2.imread(path_happy + '/' + i)
  img = cv2.resize(img,(48,48))
  total_images.append(img)
  pos[j] = 2
  j = j+1
print('Loaded images from happy dataset')

path_sadness = images_path + '/sadness'         #Loading sadness dataset
sadness_img = os.listdir(path_sadness)
features.append('sadness')
for i in sadness_img:
  img = cv2.imread(path_sadness + '/' + i)
  img = cv2.resize(img,(48,48))
  total_images.append(img)
  pos[j] = 3
  j = j+1
print('Loaded images from sadness dataset')

path_surprise = images_path + '/surprise'         #Loading surprise dataset
surprise_img = os.listdir(path_surprise)
features.append('surprise')
for i in surprise_img:
  img = cv2.imread(path_surprise + '/' + i)
  img = cv2.resize(img,(48,48))
  total_images.append(img)
  pos[j] = 4
  j = j+1
print('Loaded images from surprise dataset')

feature_count = len(os.listdir(images_path))
images = np.array(total_images).astype('float32')
images = images/255

from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
Y = np_utils.to_categorical(pos, feature_count)        #to convert array of labeled data to one-hot vector

x,y = shuffle(images,Y, random_state=2)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)        #Splitting the data into training and testing in the ratio 0.8:0.2
x_test=X_test

print(feature_count)

from keras.models import Sequential             #Importing the libraries required for creating the model
from keras.layers import Dense 
from keras.layers import Activation 
from keras.layers import Dropout 
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization

def create_model():
    input_shape=(48,48,3)

    #Instantiation
    AlexNet = Sequential()

    #1st Convolutional Layer
    AlexNet.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11,11), strides=(4,4), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    #2nd Convolutional Layer
    AlexNet.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    #3rd Convolutional Layer
    AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))

    #4th Convolutional Layer
    AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))

    #5th Convolutional Layer
    AlexNet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    #Passing it to a Fully Connected layer
    AlexNet.add(Flatten())
    # 1st Fully Connected Layer
    AlexNet.add(Dense(4096, input_shape=(32,32,3,)))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    AlexNet.add(Dropout(0.4))

    #2nd Fully Connected Layer
    AlexNet.add(Dense(4096))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    #Add Dropout
    AlexNet.add(Dropout(0.4))

    #3rd Fully Connected Layer
    AlexNet.add(Dense(1000))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    #Add Dropout
    AlexNet.add(Dropout(0.4))

    #Output Layer
    AlexNet.add(Dense(5))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('softmax'))

    AlexNet.compile(loss = keras.losses.categorical_crossentropy, optimizer= 'adam', metrics=['accuracy'])
    
    return AlexNet

model_custom = create_model()
model_custom.summary()

from keras.utils import plot_model
plot_model(model_custom, to_file='model.png', show_shapes=True, show_layer_names=True)

from sklearn.model_selection import KFold      #Importing Kfold
kf = KFold(n_splits=2, shuffle=False)

from keras.preprocessing.image import ImageDataGenerator        #ImageDataGenerator id used to allow the use of Image data augumentation

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,horizontal_flip=True, fill_mode="nearest")

epoch = 50  #Initializing epochs and batch size. Epoch size is large because we have used Earlystopping during training which can halt the training process if required
Batch_size = 8

from keras.callbacks import ModelCheckpoint         #Importing the libraries required for training the dataset
from keras.callbacks import EarlyStopping 
from keras.callbacks import ModelCheckpoint
scores_loss = []
scores_acc = []
k_no = 0
for train_index, test_index in kf.split(x):
    X_Train_ = x[train_index]
    Y_Train = y[train_index]
    X_Test_ = x[test_index]
    Y_Test = y[test_index]

    file_path = "/"+str(k_no)+".hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=1, save_best_only=True, mode='min')         #ModelCheckpoint saves the model and weight at certain points where good performance is observed    
    early = EarlyStopping(monitor="loss", mode="min", patience=8)           #It stops the training of the model when the performance ceases to improve

    callbacks_list = [checkpoint, early]

    model = create_model()
    model_data = model.fit(aug.flow(X_Train_, Y_Train), batch_size = Batch_size, epochs=epoch,validation_data=(X_Test_, Y_Test), callbacks=callbacks_list, verbose=1)        #Fitting the model onto the dataset
    model.load_weights(file_path)
    score = model.evaluate(X_Test_,Y_Test, verbose=1)
    scores_loss.append(score[0])
    scores_acc.append(score[1])
    k_no+=1

value_min = min(scores_loss)
value_index = scores_loss.index(value_min)

model.load_weights("/"+str(value_index)+".hdf5")

optimum_model = model

score = optimum_model.evaluate(X_test, y_test, verbose=1)         #Evaluating the model using x_test and y_test
print('Test Loss is = ', score[0])
print('Test accuracy is = ', score[1])

test_image = X_test[0:1]
print (test_image.shape)

print(optimum_model.predict(test_image))
print(optimum_model.predict_classes(test_image))
print(y_test[0:1])

y_pred = optimum_model.predict(X_test)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline                   
#Plotting train vs test for loss and accuracy

train_loss=model_data.history['loss']
val_loss=model_data.history['val_loss']
train_acc=model_data.history['accuracy']
val_acc=model_data.history['val_accuracy']

epochs = range(len(train_acc))

plt.plot(epochs,train_loss,'r', label='training loss')
plt.plot(epochs,val_loss,'g', label='validation loss')
plt.title('training loss vs validation loss')
plt.legend()
plt.figure()

plt.plot(epochs,train_acc,'r', label='training accuracy')
plt.plot(epochs,val_acc,'g', label='validation accuracy')
plt.title('training accuracy vs validation accuracy')
plt.legend()
plt.figure()

optimum_model.save_weights('model_weights.h5')       #Saving the models as hdf5 files
optimum_model.save('model_keras.h5')



from sklearn.metrics import confusion_matrix                 #Plotting the confusion matrix for the testing dataset
results = optimum_model.predict_classes(X_test)
conf = confusion_matrix(np.where(y_test == 1)[1], results)
feature_labels = ['anger','fear','happy','sadness','surprise']
confusion = pd.DataFrame(conf, index = feature_labels,columns = feature_labels)
plt.figure(figsize = (5,5))
sns.heatmap(confusion, annot = True,cbar=False,linewidth=2,fmt='d')
plt.title('Emotions')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

from IPython.display import clear_output, Image                 #Playing a video and recognizing the emotions from the different frames present in it
import base64
 

def arrayShow (imageArray):
    ret, png = cv2.imencode('.png', imageArray)
    encoded = base64.b64encode(png)
    return Image(data=encoded.decode('ascii'))

import time; 

cap = cv2.VideoCapture('video4.mp4')              #Read the video file

count = 0

while cap.isOpened():

    ret, frame = cap.read()

    img_data_list3=[]

    if frame is None:
      break

    input_img_resize3=cv2.resize(frame,(48,48))
    img_data_list3.append(input_img_resize3)

    img_data2 = np.array(img_data_list3)
    img_data2 = img_data2.astype('float32')
    img_data2 = img_data2/255

    detected = features[optimum_model.predict_classes(img_data2)[0]] 

    cv2.putText(frame,detected,(1, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)            #Print the  emotion on the frame itself

    bigger = cv2.resize(frame, (300, 300))

    img = arrayShow(bigger)
    display(img)

    time.sleep(1)
    clear_output(wait=True)

    if ret:
        
        count += 25  #Since video is  25 fps
        cap.set(1, count)
    else:
        cap.release()
        break

import numpy as np                    #Using the model to find the emotion present in an image after uploading 
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
 
  # predicting images
  path = '/content/' + fn
  img = image.load_img(path, target_size=(48,48))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = optimum_model.predict(images)

print('The emotion is - ')
print(features[optimum_model.predict_classes(images)[0]] )

import keras                    #Using the saved model
from keras.models import load_model
from keras.models import Sequential
model = load_model('model_keras (2).h5')
model.load_weights('model_weights (2).h5')

import numpy as np                  #Testing an upoaded image with the loaded dataset
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
 
  # predicting images
  path = '/content/' + fn
  img = image.load_img(path, target_size=(48,48))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images)

print('The emotion is - ')
features = ['anger','fear','happy','sadness','surprise']
print(features[model.predict_classes(images)[0]] )

from IPython.display import clear_output, Image                 #Playing a video and recognizing the emotions from the different frames present in it
import base64
 

def arrayShow (imageArray):
    ret, png = cv2.imencode('.png', imageArray)
    encoded = base64.b64encode(png)
    return Image(data=encoded.decode('ascii'))

import time; 

cap = cv2.VideoCapture('video4.mp4')              #Read the video file

count = 0

while cap.isOpened():

    ret, frame = cap.read()

    img_data_list3=[]

    if frame is None:
      break

    input_img_resize3=cv2.resize(frame,(48,48))
    img_data_list3.append(input_img_resize3)

    img_data2 = np.array(img_data_list3)
    img_data2 = img_data2.astype('float32')
    img_data2 = img_data2/255

    detected = features[model.predict_classes(img_data2)[0]] 

    cv2.putText(frame,detected,(1, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)            #Print the  emotion on the frame itself

    bigger = cv2.resize(frame, (300, 300))

    img = arrayShow(bigger)
    display(img)

    time.sleep(1)
    clear_output(wait=True)

    if ret:
        
        count += 25  #Since video is  25 fps
        cap.set(1, count)
    else:
        cap.release()
        break

