from keras.preprocessing.image import img_to_array
from keras.models import load_model
from utils.captchahelper import preprocess
from imutils import contours, paths
import imutils
import numpy as np
import cv2

modelPath   = "lenet.hdf5"
imagesPath = "downloads"

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(modelPath)

# randomly sample a few of the input images
imagesPaths = list(paths.list_images(imagesPath))
imagesPaths = np.random.choice(imagesPaths, size=(10,), replace=False)

# loop over the image paths
for imagePath in imagesPaths:
    # load the image and convert it to grayscale
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # pad the image
    gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_REPLICATE)

    # threshold the image to reveal the digits
    thresh = cv2.threshold(gray, 0, 255, 
                                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # find contours in the image, keeping only the four largest ones,
    # then sort them from left-to-right
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
    cnts = contours.sort_contours(cnts)[0]

    # initialize the output image as a "grayscale" image with 3
    # channels along with the output predictions
    output = cv2.merge([gray] * 3)
    predictions = []

    # loop over the contours
    for c in cnts:
        # compute the bounding box for the contour then extract the digit
        (x, y, w, h) = cv2.boundingRect(c)
        roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]

        # pre-process the ROI and classify it then classify it
        roi = preprocess(roi, 28, 28)
        roi = np.expand_dims(img_to_array(roi), axis=0)
        pred = model.predict(roi).argmax(axis=1)[0] + 1
        predictions.append(str(pred))

        # draw the prediction on the output image
        cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        cv2.putText(output, str(pred), (x - 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        
        # show the output image
        print("[INFO] captcha: {}".format("".join(predictions)))
        cv2.imshow("Output", output)
        cv2.waitKey()