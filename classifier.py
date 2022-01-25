#All the modules

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#Python Imaging Library (PIL) - external library adds support for image processing capabilities
from PIL import Image
import PIL.ImageOps
import os,ssl

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

print(pd.Series(y).value_counts())

classes = ['0', '1', '2','3', '4','5', '6', '7', '8','9']
nclasses = len(classes)

x_train,x_test,y_train,y_test = train_test_split(X,y,random_state = 9,train_size = 14600,test_size = 3400)
x_trainscale = x_train/255.0
x_testscale = x_test/255.0

clf = LogisticRegression(solver="saga",multi_class="multinomial").fit(x_trainscale , y_train)

def get_prediction(image):
    im_pil = Image.open(image)
    iBW = im_pil.convert('L')
    Ibw_resize = IBW.resize( (28,28) , Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(Ibw_resize,pixel_filter)
    I_scale = np.clip(Ibw_resize-min_pixel,0,255)
    max_pixel = np.max(Ibw_resize)
    final_scaleImg = np.asarray(I_scale)/max_pixel
    test_sample = np.array(final_scaleImg).reshape(1,784)
    test_pred = clf.predict(test_sample)
    return test_pred[0]

