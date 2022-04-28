from imutils import paths
import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn.metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers import Dropout

def loading_images(Link):
    img_dataset = []
    img_labels = []
    images_Paths = list(paths.list_images(Link))
    for address in images_Paths:
        img = cv2.imread(address)
        img = cv2.resize(img,(32,32))
        Label = address.split(os.path.sep)[6]
        img = img.astype('float32')
        img = img/255.0
        img = img.flatten()
        img_dataset.append(img)
        img_labels.append(Label)
        
    img_dataset_array = np.asarray(img_dataset)
    img_labels_array = np.asarray(img_labels)
    return img_dataset_array, img_labels_array
    
Link = r"D:\AI_Faradars\Deep_Learning_Projects\Car_Truck_Classification\train"
X , L = loading_images(Link)

#X_train, X_test, y_train, y_test = train_test_split(X, L, test_size = 0.2 , random_state=1)

all_labels = []
for i in L:
    if i=='Mask':
        all_labels.append(1)
    else:
        all_labels.append(0)

L = all_labels
L = to_categorical(L)

X_train, X_test, y_train, y_test = train_test_split(X, L, test_size = 0.2 , random_state=1)
    
model = Sequential()
model.add(Dense(300, activation = 'relu', input_dim = 32*32*3 ))
#model.add(Dropout(0.25))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))

model.summary()

model.compile(optimizer = 'SGD', loss = 'categorical_crossentropy', metrics = ['accuracy'])

h = model.fit(X_train, y_train, epochs = 50, batch_size = 16, validation_data=(X_test, y_test))

loss, acc = model.evaluate(X_test,y_test, batch_size = 16)

plt.plot(h.history['accuracy'], color = 'blue', label = 'accuracy')
plt.plot(h.history['val_accuracy'], color = 'green', label = 'val_accuracy')
plt.plot(h.history['loss'], color = 'red', label = 'loss')
plt.plot(h.history['val_loss'], color = 'orange', label = 'val_loss')
plt.title('The accuracy / loss of The model')
plt.xlabel('epochs')
plt.ylabel('accuracy_loss / val_accuracy_loss')
plt.legend()
plt.show()

model.save("MLP_mask.h5")



        

img_labels = to_categorical(img_labels)


