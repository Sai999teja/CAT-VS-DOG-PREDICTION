import os

import cv2

import matplotlib.pyplot as plt

import numpy as np

DATA_DIR = r'C:\Users\satya\Downloads\dogs-cats-images\dog vs cat\dataset\training_set'

CATEGORIES = ['cats','dogs']


#displaying single image

for i in CATEGORIES:

    for img in os.listdir(os.path.join(DATA_DIR,i)):
        path = os.path.join(DATA_DIR, i)
        img_array = cv2.imread(os.path.join(path,img),0)
        plt.imshow(img_array)
        plt.show()
        break
    break

IMG_SIZE = 100

new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))

plt.imshow(new_array)

plt.show()

TRAINING_DATA = []

def create_training_data():
    for i in CATEGORIES:                                    #iterating for cats 1st and dogs 2nd
        path = os.path.join(DATA_DIR,i)
        for img in os.listdir(path):                        #iterating to each image
            try:                                            #to pass over corrupt images
                array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                array = cv2.resize(array,(IMG_SIZE,IMG_SIZE))
                classlab = CATEGORIES.index(i)
                TRAINING_DATA.append([array,classlab])
            except Exception as e:
                pass

create_training_data()


# shuffling the data as it all the first 4000 labelled 0 and the remaining are 1

import random

random.shuffle(TRAINING_DATA)



X = []
y = []
for features,labels in TRAINING_DATA:
    X.append(features)
    y.append(labels)

X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)

print(X.shape)

y = np.array(y)

import pickle

pickle_out = open(r'C:\Users\satya\Downloads\dogs-cats-images\dog vs cat\dataset\X.pickle','wb')
pickle.dump(X,pickle_out)
pickle_out.close()


pickle_out = open(r'C:\Users\satya\Downloads\dogs-cats-images\dog vs cat\dataset\y.pickle','wb')
pickle.dump(y,pickle_out)
pickle_out.close()



