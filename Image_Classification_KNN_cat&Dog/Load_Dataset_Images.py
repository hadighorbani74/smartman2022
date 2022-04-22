from imutils import paths
import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from joblib import dump,load
import matplotlib.pyplot as plt


def loading_images(Link):
    img_dataset = []
    img_labels = []
    images_Paths = list(paths.list_images(Link))
    for address in images_Paths:
        img = cv2.imread(address,cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img,(32,32))
        Label = int(address.split(os.path.sep)[3])
        img = img.astype('float32')
        img = img/255.0
        img = img.flatten()
        img_dataset.append(img)
        img_labels.append(Label)
    img_dataset_array = np.asarray(img_dataset)
    img_labels_array = np.asarray(img_labels)
    return img_dataset_array, img_labels_array 


    
Link = "D:\AI_Faradars\Dataset"
X , L = loading_images(Link)


X_train, X_test, y_train, y_test = train_test_split(X, L, test_size = 0.2)

model = KNeighborsClassifier(n_neighbors = 7)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

for i in range(1,10):
    model = KNeighborsClassifier(n_neighbors = i)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"i : {i} , score : {score}")

dump(model, "cat&dog.joblib")






