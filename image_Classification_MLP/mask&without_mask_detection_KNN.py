from imutils import paths
import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from joblib import dump,load
import matplotlib.pyplot as plt
import sklearn.metrics


def loading_images(Link):
    img_dataset = []
    img_labels = []
    images_Paths = list(paths.list_images(Link))
    for address in images_Paths:
        img = cv2.imread(address)
        img = cv2.resize(img,(64,64))
        Label = address.split(os.path.sep)[6]
        img = img.astype('float32')
        img = img/255.0
        img = img.flatten()
        img_dataset.append(img)
        img_labels.append(Label)
        
    img_dataset_array = np.asarray(img_dataset)
    img_labels_array = np.asarray(img_labels)
    return img_dataset_array, img_labels_array
    
Link = r"D:\AI_Faradars\python_codes\Mask_withoutMask\New Masks Dataset"
X , L = loading_images(Link)


X_train, X_test, y_train, y_test = train_test_split(X, L, test_size = 0.2 , random_state=1)
model = KNeighborsClassifier(n_neighbors = 7)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
y_pred = model.predict(X_test)

sklearn.metrics.plot_confusion_matrix(model, X_test, y_test)
sklearn.metrics.plot_roc_curve(model, X_test, y_test)
print(sklearn.metrics.classification_report(y_test, y_pred))


dump(model, "mak_and_wmask.joblib")