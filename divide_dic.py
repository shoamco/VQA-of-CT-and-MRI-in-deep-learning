
import pandas as pd
import os
import numpy as np
from os.path import isdir
from os import makedirs
import shutil
from pandas import ExcelWriter
import warnings
warnings.filterwarnings("ignore")


from sklearn.utils import resample

def saveDataInExcel(mriData,ctData,name_file):
   writer = ExcelWriter('FinelFiles/'+name_file+'.xlsx')
   mriData.to_excel(writer, 'MRI', index=False)
   ctData.to_excel(writer, 'Ct', index=False)
   writer.save()


def balance_data(df_majority,df_minority):
   n_samples=len(df_majority)
   print(n_samples)
   # Upsample minority class
   df_minority_upsampled = resample(df_minority,
                            replace=True,  # sample with replacement
                            n_samples=n_samples,# to match majority class
                           random_state=50

                            )  # reproducible results

   # Combine majority class with upsampled minority class
   df_upsampled = pd.concat([df_majority, df_minority_upsampled])

   # Display new class counts
   # print(df_upsampled.balance.value_counts())
   return df_upsampled

# The function get images of ct mri,Makes a balance,And returns the data and the label
def LabelAndBalance(MriImages, CtImages):
   MriImages=MriImages.to_frame()
   CtImages=CtImages.to_frame()
   MriImages['balance'] = np.full(len(MriImages), 1)  # adding a label for each rows ('1'=mri)
   CtImages['balance'] = np.full(len(CtImages), 0)


   df_majority,df_minority = (MriImages,CtImages) if len(MriImages) >= len(CtImages) else (CtImages, MriImages)
   # print(df_majority)
   print(' df_majority: '+str(len(df_majority))+' ,df_minority:' +str(len(df_minority)))

   dfBalance = balance_data(df_majority,df_minority)

   DataImages = ["images\Train-images\\" + row['Images'] + ".jpg" for index, row in dfBalance.iterrows()]
   Label = [row['balance'] for index, row in dfBalance.iterrows()]

   return (DataImages, Label)

def open_data():
   df = pd.read_csv('InputFiles/dataset.csv',
                names=['Images', 'Questions', 'Answers'])  # open csv file and rename columns
   VQAM = pd.read_csv('InputFiles/VQAM.csv',
                  names=['Images', 'Questions', 'Answers'])  # open csv file and rename columns

   # dictionary of replaceable words
   replace_dict = {"magnetic resonance imaging": "mri",
               "mri scan": 'mri',
               "MRI": "mri",
               # " mri": "mri",
               # "mri ": "mri", " mri ": "mri",

               "shows": "show",
               "reveal": "show",
               "demonstrate": "show",
               "CT": "ct",
               # " ct": "ct"," ct ": "ct","ct ": "ct",
               "ct scan": "ct",
               "does": "", "do ": "", "the": "",
               # " a ":' ',' is ':' ',
               }
   df.replace(to_replace=replace_dict, inplace=True, regex=True)  # replace word
   VQAM.replace(to_replace=replace_dict, inplace=True, regex=True)  # replace word

###################Only mri ct vqa ##################
   # df= pd.concat([df,VQAM])
   # MriData= df[(df['Answers'].str.contains('mri')) == True]
   MriData = df[(df['Questions'].str.contains(' mri |mri | mri') | df['Answers'].str.contains(' mri |mri | mri')) == True]  # get all the Mri releted rows

   CtData = df[((df['Questions'].str.contains(' ct |ct | ct') | df['Answers'].str.contains(' ct |ct | ct'))) == True]
   # print(MriData)


   sizeMri= len(MriData)
   sizeCt= len(CtData)
##############divide all vqa to 3 group: train:80% valid 10% test :10%################

   MriTrainData =MriData[:int(sizeMri * 0.8)]
   CtTrainData = CtData[:int(sizeCt * 0.8)]

   MriValidData = MriData[int(sizeMri * 0.8):int(sizeMri * 0.9)]
   CtValidData = CtData[int(sizeCt * 0.8):int(sizeCt * 0.9)]

   MriTestData = MriData[int(sizeMri * 0.9):]
   CtTestData = CtData[int(sizeCt * 0.9):]

   ##########save in excel#########

   saveDataInExcel(MriTrainData,CtTrainData,"VQA_TrainingSet")
   saveDataInExcel(MriValidData,CtValidData,"VQA_ValidSet")
   saveDataInExcel(MriTrainData,CtTrainData,"VQA_TestSet")

#########  balence  ################
   TrainData,TrainLabel=LabelAndBalance(MriTrainData['Images'],CtTrainData['Images'])
   validData,validLabel=LabelAndBalance(MriValidData['Images'],CtValidData['Images'])

   print(len(TrainData))
   print(len(TrainLabel))
   print("____________________")
   print(len(validData))
   print(len(validLabel))

   ImagesMriTest = ["images\Train-images\\" + img + ".jpg"  for img in MriTestData['Images']]
   ImagesCtTest = ["images\Train-images\\" + img + ".jpg"  for img in CtTestData['Images']]

   testData = np.concatenate((ImagesMriTest, ImagesCtTest), axis=0, out=None)
   y = np.full((len(ImagesMriTest)), 0)
   y1 = np.full((len(ImagesCtTest)), 1)
   testLabel = np.concatenate((y, y1), axis=0, out=None)


   SizeTrain = len(TrainData)
   SizeValid = len(validData)
   SizeTest = len(testData)

   return (TrainData, TrainLabel, validData, validLabel, SizeTrain, SizeValid, SizeTest, testData, testLabel)


# Divide images into folders of train valid test
# In which each folder has a partition of CT MRI
# (To be able to use the model of keras)

#
def Divide_images_into_folders():
   TrainData, TrainLabel, validData, validLabel, SizeTrain, SizeValid, SizeTest, testData, testLabel = open_data()
   makedirs('data')

   train_folder = 'data/train'
   valid_folder = "data/valid"
   test_folder = 'data/test'





   if isdir(train_folder):  # if directory already exists
      shutil.rmtree(train_folder)
   if isdir(valid_folder):  # if directory already exists
      shutil.rmtree(valid_folder)
   if isdir(test_folder):  # if directory already exists
      shutil.rmtree(test_folder)
   makedirs(train_folder + '/ct/')
   makedirs(train_folder + '/mri/')
   makedirs(valid_folder + '/ct/')
   makedirs(valid_folder + '/mri/')
   makedirs(test_folder + '/ct/')
   makedirs(test_folder + '/mri/')

   for f, i in zip(TrainData, TrainLabel):##train folder
      basename = os.path.basename(f)
      head, tail = os.path.splitext(basename)

      if i == 0:
         dst_dir=train_folder + '/mri/'
         dst_file = os.path.join(dst_dir, basename)
         if not os.path.exists(dst_file):
             shutil.copy2(f, dst_dir)
         else:
            count = 0
            dst_file1=dst_file
            while os.path.exists(dst_file1):
               count += 1
               dst_file1 = os.path.join(dst_dir, '%s-%d%s' % (head, count, tail))
            # print 'Renaming %s to %s' % (file, dst_file)
            os.rename(dst_file, dst_file1)
            shutil.copy2(f, dst_dir)

      else:
         dst_dir = train_folder + '/ct/'
         dst_file = os.path.join(dst_dir, basename)
         if not os.path.exists(dst_file):
            shutil.copy2(f, dst_dir)
         else:
            count = 0
            dst_file1 = dst_file
            while os.path.exists(dst_file1):
               count += 1
               dst_file1 = os.path.join(dst_dir, '%s-%d%s' % (head, count, tail))
            # print 'Renaming %s to %s' % (file, dst_file)
            os.rename(dst_file, dst_file1)
            shutil.copy2(f, dst_dir)

   for f, i in zip(validData, validLabel):##valid folder

      basename = os.path.basename(f)
      if i == 0:
         dst_dir = valid_folder + '/mri/'
         dst_file = os.path.join(dst_dir, basename)
         if not os.path.exists(dst_file):
            shutil.copy2(f, dst_dir)
         else:
            count = 0
            dst_file1 = dst_file
            while os.path.exists(dst_file1):
               count += 1
               dst_file1 = os.path.join(dst_dir, '%s-%d%s' % (head, count, tail))

            os.rename(dst_file, dst_file1)
            shutil.copy2(f, dst_dir)
      else:
         dst_dir = valid_folder + '/ct/'
         dst_file = os.path.join(dst_dir, basename)
         if not os.path.exists(dst_file):
            shutil.copy2(f, dst_dir)
         else:
            count = 0
            dst_file1 = dst_file
            while os.path.exists(dst_file1):
               count += 1
               dst_file1 = os.path.join(dst_dir, '%s-%d%s' % (head, count, tail))

            os.rename(dst_file, dst_file1)
            shutil.copy2(f, dst_dir)
   for f, i in zip(testData, testLabel):##test folder
      if i == 0:
         shutil.copy2(f, test_folder + '/mri/')
      else:
         shutil.copy2(f, test_folder + '/ct/')


# Data pre-processing and data augmentation
if not os.path.exists('data'):
   Divide_images_into_folders()