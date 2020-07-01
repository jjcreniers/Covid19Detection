# USAGE
# python train.py --dataset dataset

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
from sklearn.utils.class_weight import compute_class_weight

import talos
import pandas as pd

from talos.utils import lr_normalizer
from tensorflow.keras.losses import binary_crossentropy

import random as python_random
import tensorflow as tf
from talos import Restore
from talos import Deploy
from talos.metrics.keras_metrics import f1score
import keras_metrics as km

from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score


# seeds:
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(37)
python_random.seed(1254)
tf.random.set_seed(89)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="covid19.model",
	help="path to output loss/accuracy plot"),
ap.add_argument("-i", "--inputModel", type=str, default ="None", 
  help ="path to saved input model.")
args = vars(ap.parse_args())

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-3
EPOCHS = 25
BS = 32

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
	# extract the class label from the filename
	label = imagePath.split(os.path.sep)[-2]

	# load the image, swap color channels, and resize it to be a fixed
	# 224x224 pixels while ignoring aspect ratio
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))

	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)

# convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 255]
data = np.array(data) / 255.0
labels = np.array(labels)

# perform one-hot encoding on the labels: FIRST ENTRY IS COVID, SECOND IS NORMAL
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)


#### CLASS WEIGHT GENERATION ####

y_integers = np.argmax(labels, axis=1)
class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights = dict(enumerate(class_weights))

# partition the data into training and testing splits using 85% of
# the data for training and the remaining 15% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.1, stratify=labels, random_state=42)

# take 12.5% of the training data for validation
(trainX, valX, trainY, valY) = train_test_split(trainX, trainY, test_size=(1/9), random_state=42) # 1/9 x 0.9 = 0.1


p = {'lr': [0.0005,0.0001, 0.001],
     'batch_size': [16, 32, 64],
     'epochs': [25],
     'dropout': [0.50],
     'optimizer': [Adam, SGD],
     'loss': ['binary_crossentropy'],
     'last_activation': ['softmax']}
         
#### LOAD IN MODEL ####
if (args["inputModel"] == "None"):

  # load the VGG16 network, ensuring the head FC layer sets are left
  # off
  baseModel = DenseNet121(
    include_top=False,
    weights= 'imagenet',
    input_tensor=Input(shape=(224, 224, 3))
)
else: 
  baseModel = load_model(args["inputModel"])

def transferlearn_model(trainX, trainY, valX, valY, params):

# initialize the training data augmentation object
    trainAug = ImageDataGenerator(
        rotation_range=15,
        fill_mode="nearest")

    # PRINT SUMMARY OF MODEL
    # baseModel.summary()
    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel) 
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(64, activation="relu")(headModel)
    headModel = Dropout(params['dropout'])(headModel)
    headModel = Dense(2, activation="softmax")(headModel)

    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)

    # loop over all layers in the base model and freeze them so they will
    # not be updated during the first training process
    for layer in baseModel.layers:
        layer.trainable = False

    # compile our model
    print("[INFO] compiling model...")
    opt = params['optimizer'](lr=params['lr'], decay=params['lr'] / params['epochs'])
    model.compile(loss="binary_crossentropy", optimizer=opt,
        metrics=["accuracy", km.binary_f1_score()])

    # train the head of the network
    print("[INFO] training head...")
    out = model.fit_generator(
        trainAug.flow(trainX, trainY, batch_size=params['batch_size']),
        steps_per_epoch=len(trainX) // params['batch_size'],
        validation_data=(valX, valY),
        validation_steps=len(valX) // params['batch_size'],
        epochs=params['epochs'],
    class_weight = d_class_weights,
    verbose=0)
    return out, model

scan_object = talos.Scan(trainX,
                         trainY, 
                         params=p,
                         x_val = valX,
                         y_val = valY,
                         model=transferlearn_model,
                         experiment_name='drive/My Drive/ComputerVision/keras-covid-19/imagenetscan_adam_nodrop',
                         seed = 42)

# talos.Deploy(scan_object=scan_object, model_name='imagenetvebosedeploy2', metric='val_accuracy');
