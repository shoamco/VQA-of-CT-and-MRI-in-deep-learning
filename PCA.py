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
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage.feature import daisy
from skimage.feature import hog
from skimage import data, exposure
import random


def CALC_SHOW_PCA_3D(Data, STRING_MODEL):
    pca = PCA(n_components=3)
    DataPCA = pca.fit(Data).transform(Data)
    target_names = ['MRI', 'CT']
    group1 = np.full((len(ImagesOfMri)), 0)#Mri Group=0
    group2 = np.full((len(ImagesOfCt)), 1)#ct Group=0
    Group = np.concatenate((group1, group2), axis=0, out=None)#all Groups
    x = [x for (x ,y , z) in DataPCA]
    y = [y for (x ,y , z) in DataPCA]
    z=[z for (x ,y , z) in DataPCA]

    fig = pyplot.figure()
    ax = Axes3D(fig)
    colors = ['red', 'blue']
    lw = 2

    for color, i, target_name in zip(colors, [0, 1], target_names):#Draw a 3D graph by the colored groups
      ax.scatter(DataPCA[Group == i, 0], DataPCA[Group == i, 1], DataPCA[Group == i, 2], color=color, alpha=.8, lw=lw, label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of mri/ct '+STRING_MODEL)

    pyplot.show()

def img_to_histogram(images):
    # create an array of histogram for all the images
    img_to_hist = [cv2.calcHist([img], [0], None, [256], [0, 256]) for img in images]
    # Calculation of a normalized histogram
    img_to_hist = [norm_hist(hist) for hist in img_to_hist]
    return img_to_hist
def calc_dsy(images):
    # img_to_2d=two_dim(images)
    # print(img_to_2d)
    img_to_dsy = [feature.daisy(img, step=15, radius=15, rings=3, histograms=8, orientations=4, normalization='l1', sigmas=None, ring_radii=None, visualize=False) for img in images]


    return img_to_dsy
#
def norm_hist(hist):
    img_to_hist = cv2.normalize(hist, hist)
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


def concatenate_mri_ct(mri_arr, ct_arr):
    return np.concatenate((mri_arr, ct_arr), axis=0, out=None)




df = pd.read_csv('InputFiles/dataset.csv', names=['Images', 'Questions', 'Answers'])  # open csv file and rename columns

# # dictionary of replaceable words
replace_dict = {"magnetic resonance imaging": "mri",
                "mri scan": 'mri',
                "MRI": "mri",
                "shows": "show",
                "reveal": "show",
                "demonstrate": "show",
                "CT": "ct",
                "ct scan": "ct",
                "does": "", "do ": "", "the": "",
                # " a ":' ',' is ':' ',
                }
df.replace(to_replace=replace_dict, inplace=True, regex=True)  # replace word

ImagesOfMri = df[(~df['Questions'].str.contains('mri|ct') & df['Questions'].str.contains('what') & df[
    'Answers'].str.contains('mri')) == True]['Images']
ImagesOfCt = df[(~df['Questions'].str.contains('mri|ct') & df['Questions'].str.contains('what') & df[
    'Answers'].str.contains('ct')) == True]['Images']
# loading all the image that it's answers contain mri or ct
ImagesOfMri = [cv2.imread("images\Train-images\\" + img + ".jpg") for img in ImagesOfMri]
ImagesOfCt = [cv2.imread("images\Train-images\\" + img + ".jpg") for img in ImagesOfCt]



# *************HISTOGRAM********************
# mri_hist = two_dim(img_to_histogram(ImagesOfMri))
# ct_hist = two_dim(img_to_histogram(ImagesOfCt))
#
# data_hist = concatenate_mri_ct(mri_hist, ct_hist)
#
# CALC_SHOW_PCA_3D(data_hist, "HISTOGRAM")

# *************DAISY********************
# mri_hist =calc_dsy(ImagesOfMri)
# ct_hist = two_dim(calc_dsy(ImagesOfCt))

# data_hist = concatenate_mri_ct(mri_hist, ct_hist)
#
# CALC_SHOW_PCA_3D(data_hist, "HISTOGRAM")
# img=cv2.imread("images\Valid-images\\SJA-7-347-g002.jpg")
# # img_to_dsy =daisy(img, step=15, radius=15, rings=3, histograms=8, orientations=4, normalization='l1', sigmas=None, ring_radii=None, visualize=False)
# img=two_dim(img)
# descs_img=daisy(img, step=4, radius=15, rings=3, histograms=8, orientations=8, normalization='l1', sigmas=None, ring_radii=None, visualize=False)
#
#
#
# print(descs_img)