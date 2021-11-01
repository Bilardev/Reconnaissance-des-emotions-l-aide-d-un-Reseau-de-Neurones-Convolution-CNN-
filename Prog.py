# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 14:52:22 2021

@author: baziz
"""
#Importation des Librairies 
import sys, os
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils

#Chargement de dataset (je de Données)
df=pd.read_csv('fer2013.csv')

#28709 EXEMPLES POUR ENTRAINEMENT 
#2589 EXEMPLES POUR TEST 

X_train,train_y,X_test,test_y=[],[],[],[]

#Traitement et repartition de données 
for index, row in df.iterrows():
    val=row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
           X_train.append(np.array(val,'float32'))
           train_y.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
           X_test.append(np.array(val,'float32'))
           test_y.append(row['emotion'])
    except:
        print(f"error occured at index :{index} and row:{row}")

num_features = 64
num_labels = 7
batch_size = 64
epochs = 60
width, height = 48, 48


X_train = np.array(X_train,'float32')
train_y = np.array(train_y,'float32')
X_test = np.array(X_test,'float32')
test_y = np.array(test_y,'float32')


from keras.utils.np_utils import to_categorical

train_y=np_utils.to_categorical(train_y, num_classes=num_labels)
test_y=np_utils.to_categorical(test_y, num_classes=num_labels)


#Normalisation des données (- Moy/ ecart-t)
X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)

X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)


#Construction de notre model 

model = Sequential()

#couchesb d'entrée 
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:])))
model.add(Conv2D(64,kernel_size= (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

#2eme couche de convolution 
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

#3eme couche de convolution 
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))


model.add(Flatten())

# Reseau de neurones completement connecté 
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

#couche de sortie 
model.add(Dense(num_labels, activation='softmax'))

#Compilation du model 
model.compile(loss=categorical_crossentropy,
              optimizer='sgd',
              metrics=['accuracy'])

#Excution du model  
model.fit(X_train, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, test_y),
          shuffle=True)

#Suvegarde de notre model 

fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")

import os
import cv2
import numpy as np 
from keras.models import model_from_json
from keras.preprocessing import image



#chargement du  model 
model = model_from_json(open("fer.json", "r").read())
model.load_weights('fer.h5')



#Evaluation du model avec les données Test 

import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix

# Précison du model 
precision = model.evaluate(X_test, test_y)
print(f"La précision du model est de {str(precision[1]*100)[:5]} %")




# test de notre model 
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#from google.colab.patches import cv2_imshow
import cv2
import matplotlib.pyplot as plt  
test_image = cv2.imread('1.jpg')
#plt.imshow(test_image)
test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
plt.imshow(test_image)
#plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
#cv2.imshow(test_image)

gray_image= cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY) 
plt.imshow(gray_image)

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
from keras_preprocessing.image import img_to_array

faces = face_cascade.detectMultiScale(gray_image,1.1,4)
for x,y,w,h in faces :
    cv2.rectangle(test_image,(x,y),(x+w, y+h), (255,0,0))
    roi_gray=gray_image[y:y+w, x:x+h]
    roi_gray= cv2.resize(roi_gray,(48,48))
    image_pixels=img_to_array(roi_gray)
    image_pixels=np.expand_dims(image_pixels, axis=0)
    image_pixels /= 255
    predictions = model.predict(image_pixels)
    max_index = np.argmax(predictions[0])
    emotions_detection = ('en colere', 'degout', 'peur', 'Heureux', 'Triste', 'Surpris', 'Neutre')
    emotions_prediction = emotions_detection[max_index]
    print (emotions_detection)
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale =1
    color = (255, 0, 0)  
    thikness = 2
    image = cv2.putText(test_image, emotions_prediction, org, font,
                        fontScale, color, thikness, cv2.LINE_AA)
    plt.imshow(image)
    
    
# Tester notre model sur une camera en temps reel 


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
from keras_preprocessing.image import img_to_array

cap=cv2.VideoCapture(0)

while True:
    ret,test_image=cap.read()# captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_image= cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    
    faces = face_cascade.detectMultiScale(gray_image,1.1,4)

    for (x,y,w,h) in faces:
        
        cv2.rectangle(test_image,(x,y),(x+w, y+h), (255,0,0))
        roi_gray=gray_image[y:y+w, x:x+h]
        roi_gray=cv2.resize(roi_gray,(48,48))
        image_pixels=img_to_array(roi_gray)
        image_pixels=np.expand_dims(image_pixels, axis=0)
        image_pixels /= 255
        predictions = model.predict(image_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])
        
    

        emotions_detection = ('en colere', 'degout', 'peur', 'Heureux', 'Triste', 'Surpris', 'Neutre')
       
        emotions_prediction = emotions_detection[max_index]
     
        print (emotions_detection)
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale =1
        color = (255, 0, 0)  
        thikness = 2
        
        cv2.putText(test_image, emotions_prediction, org, font,
                        fontScale, color, thikness, cv2.LINE_AA)

        

    resized_img = cv2.resize(test_image, (1000, 800))
    cv2.imshow('Reconnaissance d emotions',resized_img)



    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows()