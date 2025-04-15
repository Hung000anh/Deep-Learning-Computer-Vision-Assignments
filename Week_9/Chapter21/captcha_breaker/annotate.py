# This code has a bug.. It always skips all the images

from imutils import paths
import imutils
import cv2
import os

imagesPath = "downloads"
outputPath = "dataset" # where annotations are saved

# grab the image paths
imagesPaths = list(paths.list_images(imagesPath))
# initialize the dictionary of character counts
counts = {}

# loop over the image paths
for i, imagePath in enumerate(imagesPaths):
    # display an update to the user
    print("[INFO] processing image "+ str(i+1) + "/" + str(len(imagesPaths)))

    try:
        # load the image and convert it to grayscale
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # pad the image to ensure digits caught on the border 
        # of the image are retained
        gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

        # threshold the image to reveal the digits
        thresh = cv2.threshold(gray, 0, 255,
                                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # find contours in the image, keeping only the four largest ones
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
        # loop over the contours
        for cnt in cnts:
            # compute the bounding box for the contour
            (x, y, w, h) = cv2.boundingRect(c)
            # extract the digit
            roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]
            # display the character, making it larger enough for us to see
            cv2.imshow("ROI", roi)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(","):
                print("[INFO] ignoring character")
                continue
            
            # grab the key that was pressed and construct the path
            key = char(key).upper()
            dirPath = os.path.sep.join([outputPath, key])
            # if the output directory does not exist, create it
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)
            
            # write the labeled character to file
            count = counts.get(key, 1)
            p = os.path.sep.join([dirPath, "{}.png".format(str(count).zfill(6))])
            cv2.imwrite(p, roi)

            # increment the count for the current key
            counts[key] = count + 1

    except KeyboardInterrupt:
        print("[INFO] manually leaving script")
        break

    # an unknown error has occurred for this particular image
    except:
        print("[INFO] skipping image...")