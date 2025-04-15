from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD
from nn.conv import LeNet
from utils.captchahelper import preprocess
from utils.cnnhelper import plotHistory
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

datasetPath = "dataset"
numEpochs   = 15

# initialize the data and labels
data = []
labels = []

# loop over the input images
for imagePath in paths.list_images(datasetPath):
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = preprocess(image, 28, 28)
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, 
                                            random_state=42)

# convert the labels from integers to vectors
labelBinarizer = LabelBinarizer().fit(trainY)
trainY = labelBinarizer.transform(trainY)
testY  = labelBinarizer.transform(testY)

# initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=1, classes=9)
opt = SGD(lr=0.01)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
history = model.fit(trainX, trainY, validation_data=(testX, testY), 
                    batch_size=32, epochs=numEpochs, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), 
            predictions.argmax(axis=1), target_names=labelBinarizer.classes_))

# save the model to disk
print("[INFO] serializing network...")
model.save("lenet.hdf5")

plotHistory(history, numEpochs)