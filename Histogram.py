"""
Initial analysis of MRI / CT image type
Using an average histogram
(Only for questions that do not include MRI / CT)
"""

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
from sklearn.preprocessing import scale



def img_to_histogram(images):
    # loading all the image that it's answers contain mri or ct
    name_img = [cv2.imread("images\Train-images\\" + img + ".jpg") for img in images]
    # create an array of histogram for all the images
    img_to_hist = [cv2.calcHist([img], [0], None, [256], [0, 256]) for img in name_img]
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

def pca_calc(array):
    """The function accepts an array and calculates the PCA value"""
    pca = PCA(n_components=2)
    pca.fit(two_dim(array))
    PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)
    print(pca.explained_variance_ratio_)
    pc=plt.plot(pca.explained_variance_ratio_,color='b')
    plt.show()



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

# predictions.replace(to_replace=replace_dict, inplace=True, regex=True)#replace word

# what=df_ct = df[df['Questions'].str.contains('what')]#
# data=df[(~df['Questions'].str.contains('mri|ct') & df['Questions'].str.contains('what'))==True ]#
# data2=df[(~df['Questions'].str.contains('mri|ct') & df['Questions'].str.contains('what') &df['Answers'].str.contains('mri|ct') )==True ]#
# data3=df[(~df['Questions'].str.contains('mri|ct') & df['Questions'].str.contains('what') & (df['Answers'].str.contains('mri|ct')==False) )==True ]#
# writer = pd.ExcelWriter('outputFiles/ VALID questions without CT MRI.xlsx')
# data.to_excel(writer,'All Answers',index=False)
# data2.to_excel(writer,'only MRI CT in Answers',index=False)
# data3.to_excel(writer,'not MRI CT  in Answers',index=False)
#
# writer.save()


# Extracts only relevant answers

ImagesOfMri=df[(~df['Questions'].str.contains('mri|ct') & df['Questions'].str.contains('what') &df['Answers'].str.contains('mri') )==True ]['Images']
ImagesOfCt=df[(~df['Questions'].str.contains('mri|ct') & df['Questions'].str.contains('what') &df['Answers'].str.contains('ct') )==True ]['Images']

HIST=img_to_histogram(ImagesOfMri)
HIST1=img_to_histogram(ImagesOfCt)
HIST.append(HIST1)



model_of_pc_mri=img_to_histogram(ImagesOfMri)
model_of_pc_ct=img_to_histogram(ImagesOfCt)
#
# # Calculation of an avarage histogram
# avg_hist_mri=calc_avg_hist(img_to_histogram(ImagesOfMri))
# avg_hist_ct=calc_avg_hist(img_to_histogram(ImagesOfCt))
#
#
# # Calculation of a normalized  avarage histogram
# norm_img_to_hist_mri =norm_hist(avg_hist_mri)
# norm_img_to_hist_ct =norm_hist(avg_hist_ct)
#
# # Calculation of a normalized histogram of single Image
# ingValid=cv2.imread("images\Valid-images\\SJA-7-347-g002.jpg")
# hist = cv2.calcHist(ingValid, [0], None, [256], [0, 256])
# norm_img_to_hist =norm_hist(hist)
#
#
# plt.title('Comparison of histograms (mri and ct)')
# mri_patch = mpatches.Patch(color='b', label='mri')
# ct_patch = mpatches.Patch(color='g', label='ct')
# valid_patch = mpatches.Patch(color='r', label='valid')
# #
# # # Display of normalized histograms
# #
# mri=plt.plot(norm_img_to_hist_mri,color='b')
# ct=plt.plot(norm_img_to_hist_ct,color='g')
# img=plt.plot(norm_img_to_hist,color='r')
# #
# # # Logarithmic scale presentation
# # mri=plt.semilogy(norm_img_to_hist_mri,color='b')
# # ct=plt.semilogy(norm_img_to_hist_ct,color='g')
# # img=plt.semilogy(norm_img_to_hist,color='r')
# #
# # # plt.xlim([0, 256])
# # # plt.ylim([0,20000])
# #
# plt.legend(handles=[mri_patch,ct_patch,valid_patch])
# plt.show()
pca_calc(model_of_pc_mri)
pca_calc(model_of_pc_ct)



