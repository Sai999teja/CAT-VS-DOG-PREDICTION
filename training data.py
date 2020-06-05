import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
import pickle

from tensorflow.keras.layers import Conv2D,MaxPooling2D

pickle_in = open(r'C:\Users\satya\Downloads\dogs-cats-images\dog vs cat\dataset\X.pickle','rb')
X = pickle.load(pickle_in)

pickle_in = open(r'C:\Users\satya\Downloads\dogs-cats-images\dog vs cat\dataset\y.pickle','rb')
y = pickle.load(pickle_in)

print(X.shape)


#let us normalize the data

X = X/255.0

model= Sequential()

model.add(Conv2D(256,(3,3),input_shape=(100,100,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))




model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X,y,batch_size=4,epochs=10,validation_split=0.3)

model.save(r"C:\Users\satya\Downloads\dogs-cats-images\dog vs cat\dataset\mymodel.model")