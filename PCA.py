

import pandas as pd
import cv2
from cv2 import *
import numpy as np
from scipy.stats import wasserstein_distance
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.legend_handler \
import HandlerLine2D
import matplotlib.patches as mpatches
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from skimage import feature
from skimage import feature
from sklearn.preprocessing import scale
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def CALC_SHOW_PCA(X,STRING_MODEL):

    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)
    target_names = ['MRI', 'CT']
    y = np.full((len(ImagesOfMri)), 0)
    y1 = np.full((len(ImagesOfCt)), 1)
    y = np.concatenate((y, y1), axis=0, out=None)

    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))

    plt.figure()
    colors = ['navy', 'turquoise']
    lw = 2

    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of mri/ct '+STRING_MODEL)
    plt.show()


def img_to_histogram(images):

    # create an array of histogram for all the images
    img_to_hist = [cv2.calcHist([img], [0], None, [256], [0, 256]) for img in images]
    # Calculation of a normalized histogram
    img_to_hist = [norm_hist(hist) for hist in img_to_hist]
    return img_to_hist

def norm_hist(hist):
    img_to_hist=cv2.normalize(hist, hist)
    return img_to_hist

def calc_avg_hist(array_of_hist):
    """The function get a list names of relevant images end return tha avarage histogram for all images"""

    # convert list to an array
    a = np.array(array_of_hist)
    # Calculation of average histogram
    avg_hist = np.mean(a, axis=0)
    return avg_hist

def two_dim(array):
    """"The function accepts an array with more than two dimensions and returns an array with one dimension"""
    two_dim = [hist.flatten() for hist in array]
    return two_dim

def concatenate_mri_ct(mri_arr,ct_arr):
    return np.concatenate((mri_arr,ct_arr),axis=0, out=None)

def calc_canny(array_of_img):
    img_to_canny=[cv2.Canny(im,1,30,L2gradient=False)for im in array_of_img]
    return img_to_canny



df = pd.read_csv('InputFiles/dataset.csv',names=['Images', 'Questions', 'Answers'])#open csv file and rename columns

# # dictionary of replaceable words
replace_dict = {"magnetic resonance imaging":"mri",
                "mri scan":'mri',
                "MRI":"mri",
                "shows": "show",
                "reveal":"show",
                "demonstrate":"show",
                "CT":"ct",
                "ct scan":"ct",
                "does":"","do ":"","the":"",
                    # " a ":' ',' is ':' ',
                }
df.replace(to_replace=replace_dict, inplace=True, regex=True)#replace word


ImagesOfMri=df[(~df['Questions'].str.contains('mri|ct') & df['Questions'].str.contains('what') &df['Answers'].str.contains('mri') )==True ]['Images']
ImagesOfCt=df[(~df['Questions'].str.contains('mri|ct') & df['Questions'].str.contains('what') &df['Answers'].str.contains('ct') )==True ]['Images']

ImagesOfMri = [cv2.imread("images\Train-images\\" + img + ".jpg") for img in ImagesOfMri]
ImagesOfCt = [cv2.imread("images\Train-images\\" + img + ".jpg") for img in ImagesOfCt]

# loading all the image that it's answers contain mri or ct

# *************HISTOGRAM********************
mri_hist=two_dim(img_to_histogram(ImagesOfMri))
ct_hist=two_dim(img_to_histogram(ImagesOfCt))

data_hist=concatenate_mri_ct(mri_hist,ct_hist)

# CALC_SHOW_PCA(data_hist,"HISTOGRAM")


# **************CANNY************************

mri_canny=two_dim(calc_canny(ImagesOfMri))
ct_canny=two_dim(calc_canny(ImagesOfCt))
data_canny=concatenate_mri_ct(mri_canny,ct_canny)
print(len(data_canny[0]))
print(len(data_canny[50]))
[list(item) for item in mri_canny]
# norm_image = cv2.normalize(data_canny[0], data_canny[0])
print(mri_canny[:3])
# a=mri_canny[0]
#
# data_canny1=[np.r_[a, np.zeros((356577 - a.shape[0], 117300), dtype=a.dtype)] ]
# CALC_SHOW_PCA(data_canny1,"CANNY")

# edges = cv2.Canny(ImagesOfMri[5],1,30)
# # edges = filter.canny(ImagesOfMri[0], sigma=3)
# plt.subplot(121),
# plt.imshow(ImagesOfMri[5],cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()

# CALC_SHOW_PCA(data_canny,"CANNY")



