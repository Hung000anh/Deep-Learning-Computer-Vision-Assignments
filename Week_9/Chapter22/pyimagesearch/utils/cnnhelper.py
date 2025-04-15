import matplotlib.pyplot as plt
import numpy as np

def plotHistory(H, numEpochs):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, numEpochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, numEpochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, numEpochs), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, numEpochs), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()