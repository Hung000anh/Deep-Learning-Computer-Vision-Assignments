from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from nn.conv import LeNet
from imutils import paths
import imutils
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from utils.cnnhelper import plotHistory

datasetPath = "smiles"
outputModel = "lenet.hdf5"
numEpochs   = 15

# initialize the list of data and labels
data   = []
labels = []

# loop over the input images
for imagePath in sorted(list(paths.list_images(datasetPath))):
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the labels list
    label = imagePath.split(os.path.sep)[-3]
    label = "smiling" if label == "positives" else "not_smiling"
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data   = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# convert the labels from integers to vectors
labelEncoder = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(labelEncoder.transform(labels), 2)

# account for skew in the labeled data
# computes the total number of examples per class
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals
# In this case, classTotals will be an array: [9475, 3690] 
# for “not smiling” and “smiling”, respectively
# So, classWeight used to handle the class imbalance, 
# yielding the array: [1, 2.56]

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, 
                        test_size=0.20, stratify=labels, random_state=42)

# initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=1, classes=2)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# train the network
print("[INFO] training network...")
history = model.fit(trainX, trainY, validation_data=(testX, testY),
            class_weight=classWeight, batch_size=64, epochs=numEpochs, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                                target_names=labelEncoder.classes_))

# save the model to disk
print("[INFO] serializing network...")
model.save(outputModel)

plotHistory(history, numEpochs)