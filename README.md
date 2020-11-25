# Digits Classification Neural Network


![](https://raw.githubusercontent.com/twhipple/MNIST_Digits_Classification_NN/main/Images/john-barkiple-unsplash.jpg)

*My idea for a visual of a simple multi-layer perceptron Neural Network. Source: John Barkple, freeimages.com*


## Intro
This notebook is my attempt to build a basic neural networks model that can recognize hand-written digits.

The MNIST dataset is available through the scikit-learn datasets as well as other open-sourced platforms. Here you find black and white images of hand-written digits (numbers 0 through 9). This dataset consist of 60,000 examples in the training set and 10,000 examples in the test set - which are already nicely split. Plus, the digits have been size-normalized and centered in a fixed-size image.


## README Outline
* Introduction and README outline
* Repo Contents
* Libraries & Prerequisites
* Results
* Future Work
* Built With, Contributors, Authors, Acknowledgments


![](https://raw.githubusercontent.com/twhipple/MNIST_Digits_Classification_NN/main/Images/digits_sample.png)

*Sample of handwritten digits from the MNIST dataset.*


## Repo Contents
This repo contains the following:
* README.md - this is where you are now!
* Notebook.ipynb - the Jupyter Notebook containing the finalized code for this project.
* LICENSE.md - the required license information.
* dataset - the dataset can be found in the scikit-learn datasets and is not a separate file.
* CONTRIBUTING.md
* Images


## Libraries & Prerequisites
These are the libraries that I used in this project.

* import numpy as np
* import pandas as pd
* import random as rn
* Tensorflow
* import tensorflow.random as tfr
* import tensorflow.keras as keras
* from tensorflow.keras.models import Sequential, load_model
* from tensorflow.keras.layers import Dense, Dropout, Flatten
* from tensorflow.keras.layers import Conv2D, MaxPool2D, MaxPooling2D, BatchNormalization
* from tensorflow.keras import backend as K
* from tensorflow.keras.utils import to_categorical
* from tensorflow.keras.optimizers import RMSprop, Adam
* from tensorflow.keras.preprocessing.image import ImageDataGenerator
* from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
* Chart
* import matplotlib.pyplot as plt
* import matplotlib.image as mpimg
* import seaborn as sns
* from sklearn.metrics import classification_report
* from PIL import Image
* import os
* import cv2


## Conclusions
So the Convolutional Nueral Network did slightly better than the basic Multi-layer Perceptron NN. Both had pretty high valadation accuracy results even after only a couple Epochs.


## Future Work
With some help from my former instructor, Jeff Herman, I was finally able to get these simple models to run. I'm sure there are more parameters I could adjust, or add more layers to the perceptron in either model. It didn't seem like running more Epochs would make a difference, though I could change the batch size. I would also really like to see how this model works with additonal hand-written digits.


![](https://raw.githubusercontent.com/twhipple/MNIST_Digits_Classification_NN/main/Images/gary-scott-cross-connected.jpg)

*My idea of what is going on under a neural network. Source: Gary Scott, freeimages.com*


## Built With:
Jupyter Notebook
Python 3.0
scikit.learn

## Contributing
Please read CONTRIBUTING.md for details

## Authors
Thomas Whipple

## License
Please read LICENSE.md for details

## Acknowledgments
Thank you to Jeff Herman for all of your support and help with code!
Thank you to scikit-learn for the dataset.
